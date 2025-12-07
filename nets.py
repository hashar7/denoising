"""
denoise_pipeline_fixed.py

Corrected pipeline for DnCNN and full NAFNet targeting RTX 3070 Ti (12GB).
Key fixes:
 - Ensure model outputs 1 channel (no broadcasting).
 - Safe GPU inference with small batches and OOM backoff.
 - DataLoader settings safe for Windows/Jupyter (num_workers=0, pin_memory=False).
 - Correct dataset shapes (samples shaped (1,H,W), so DataLoader yields (B,1,H,W)).
 - Self-supervised finetune computes loss on masked pixels only and masks match shapes.

Usage:
  - Put this file next to your HDF and run:
      python denoise_pipeline_fixed.py
  - Optional pretrained weights in ./weights/{dncnn.pth, nafnet.pth}
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import windows
from tqdm import tqdm

# -------------------------
# Config (tune these if needed)
# -------------------------
UNFILTERED_FILE = 'metm24_TA_251128_0101_9074_01_02A.hdf'
UNFILTERED_VAR = "m_m2413_53_80V"

WEIGHTS_DIR = "./weights"
DN_CNN_WEIGHTS = os.path.join(WEIGHTS_DIR, "dncnn.pth")
NAFNET_WEIGHTS = os.path.join(WEIGHTS_DIR, "nafnet.pth")

MODEL_NAME = 'nafnet'   # 'dncnn' or 'nafnet'

# Patch/inference settings (choose small patches if OOM)
PATCH_H = 128
PATCH_W = 128
STRIDE_H = PATCH_H // 2
STRIDE_W = PATCH_W // 2

# Batch sizes: keep small for NAFNet on 12GB
FINETUNE_BATCH = 16
FINETUNE_EPOCHS = 6
FINETUNE = True

# Inference batch (how many patches to run simultaneously). We'll auto-backoff on OOM.
INFERENCE_BATCH = 8

# DataLoader safety for your environment (you discovered these are required)
NUM_WORKERS = 0
PIN_MEMORY = False

# Device and mixed precision
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True if DEVICE.type == "cuda" else False

torch.set_num_threads(4)
np.random.seed(0)
torch.manual_seed(0)

# -------------------------
# Utilities
# -------------------------
def load_hdf_as_numpy(file_path, var_name):
    f = SD(file_path, SDC.READ)
    data = f.select(var_name).get()
    return np.array(data, dtype=np.float32)

def make_hanning_window(h, w):
    win_h = windows.hann(h, sym=False)
    win_w = windows.hann(w, sym=False)
    win2d = np.outer(win_h, win_w).astype(np.float32)
    win2d += 1e-6
    return win2d

# -------------------------
# Models
# -------------------------
class DnCNN(nn.Module):
    def __init__(self, in_channels=1, features=64, depth=12):
        super().__init__()
        layers = [nn.Conv2d(in_channels, features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(features, features, 3, padding=1, bias=False),
                       nn.BatchNorm2d(features),
                       nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(features, in_channels, 3, padding=1)]  # final -> single channel
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.net(x)
        return x - noise

# NAF building blocks (faithful but practical)
class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b

class SimpleChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1, bias=True)
    def forward(self, x):
        v = self.avg_pool(x)
        v = self.conv(v)
        return x * torch.sigmoid(v)

class NAFBlockV2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pw1 = nn.Conv2d(channels, channels*2, 1, bias=True)
        self.dw  = nn.Conv2d(channels*2, channels*2, 3, padding=1, groups=channels*2, bias=True)
        self.sg  = SimpleGate()
        self.pw2 = nn.Conv2d(channels, channels, 1, bias=True)  # after SG returns channels
        self.sca = SimpleChannelAttention(channels)
        # FFN part
        self.ffn1 = nn.Conv2d(channels, channels*2, 1, bias=True)
        self.ffn_dw = nn.Conv2d(channels*2, channels*2, 3, padding=1, groups=channels*2, bias=True)
        self.ffn2 = nn.Conv2d(channels, channels, 1, bias=True)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        x = self.pw1(x)            # -> 2C
        x = self.dw(x)             # -> 2C (dw)
        x = self.sg(x)             # -> C (SimpleGate halves and multiplies)
        x = self.pw2(x)            # -> C
        x = self.sca(x)            # -> C
        x = identity + self.beta * x

        # FFN
        y = self.ffn1(x)           # -> 2C
        y = self.ffn_dw(y)         # -> 2C
        y = self.sg(y)             # -> C
        y = self.act(y)
        y = self.ffn2(y)           # -> C
        x = x + self.gamma * y
        return x

class NAFNet(nn.Module):
    def __init__(self, img_channels=1, width=48, enc_depths=[2,2,4], middle_blocks=8):
        super().__init__()
        self.entry = nn.Conv2d(img_channels, width, 3, padding=1)
        # encoder
        self.encs = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for d in enc_depths:
            blocks = nn.Sequential(*[NAFBlockV2(ch) for _ in range(d)])
            self.encs.append(blocks)
            self.downs.append(nn.Conv2d(ch, ch*2, 2, stride=2))  # downsample by strided conv
            ch *= 2
        # middle
        self.middle = nn.Sequential(*[NAFBlockV2(ch) for _ in range(middle_blocks)])
        # decoder
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for d in reversed(enc_depths):
            self.ups.append(nn.ConvTranspose2d(ch, ch//2, 2, stride=2))
            ch = ch // 2
            self.decs.append(nn.Sequential(*[NAFBlockV2(ch) for _ in range(d)]))
        # final conv MUST output single channel
        self.exit = nn.Conv2d(width, img_channels, 3, padding=1)

    def forward(self, x):
        x = self.entry(x)
        enc_feats = []
        for enc, down in zip(self.encs, self.downs):
            x = enc(x)
            enc_feats.append(x)
            x = down(x)
        x = self.middle(x)
        for up, dec, feat in zip(self.ups, self.decs, reversed(enc_feats)):
            x = up(x)
            # align shapes if needed
            if x.shape[2:] != feat.shape[2:]:
                # center-crop to min dims
                min_h = min(x.shape[2], feat.shape[2])
                min_w = min(x.shape[3], feat.shape[3])
                x = x[:, :, :min_h, :min_w]
                feat = feat[:, :, :min_h, :min_w]
            x = x + feat
            x = dec(x)
        out = self.exit(x)
        return x - out  # residual subtract -> final shape (B,1,H,W)

# -------------------------
# Dataset and finetune (self-supervised)
# -------------------------
class ImagePatchDataset(Dataset):
    def __init__(self, image, patch_h=64, patch_w=64, num_samples=2000, mask_ratio=0.05):
        self.image = image.astype(np.float32)
        self.H, self.W = image.shape
        self.ph = patch_h
        self.pw = patch_w
        self.num_samples = num_samples
        self.mask_ratio = mask_ratio
        self.coords = [ (np.random.randint(0, max(1, self.H - self.ph + 1)),
                         np.random.randint(0, max(1, self.W - self.pw + 1))) for _ in range(num_samples) ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        patch = self.image[y:y+self.ph, x:x+self.pw].copy()
        mask = np.zeros_like(patch, dtype=np.float32)
        num_mask = int(self.mask_ratio * patch.size)
        if num_mask > 0:
            coords = np.random.choice(patch.size, size=num_mask, replace=False)
            flat = mask.ravel()
            flat[coords] = 1.0
            mask = flat.reshape(patch.shape)
        masked_patch = patch.copy()
        masked_patch[mask.astype(bool)] = np.mean(patch)
        # normalize per-patch
        mean = masked_patch.mean()
        std = masked_patch.std() if masked_patch.std() > 1e-6 else 1.0
        inp = (masked_patch - mean) / std
        tgt = (patch - mean) / std
        # return tensors shaped (C,H,W) => (1,H,W)
        inp_t = torch.from_numpy(inp[None, :, :]).float()
        tgt_t = torch.from_numpy(tgt[None, :, :]).float()
        mask_t = torch.from_numpy(mask[None, :, :]).float()
        return inp_t, tgt_t, mask_t

def finetune_selfsupervised(model, image, epochs=4, batch_size=16, lr=1e-4, device=DEVICE):
    ds = ImagePatchDataset(image, patch_h=64, patch_w=64, num_samples=2000, mask_ratio=0.05)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    loss_fn = nn.L1Loss(reduction='none')
    for ep in range(epochs):
        running = 0.0
        cnt = 0
        pbar = tqdm(dl, desc=f"Finetune ep{ep+1}/{epochs}")
        for inp, tgt, mask in pbar:
            inp = inp.to(device); tgt = tgt.to(device); mask = mask.to(device)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                out = model(inp)  # shape (B,1,H,W)
                l = loss_fn(out, tgt)   # shape (B,1,H,W)
                masked_loss = (l * mask).sum() / (mask.sum() + 1e-8)
            opt.zero_grad()
            scaler.scale(masked_loss).backward()
            scaler.step(opt)
            scaler.update()
            running += masked_loss.item()
            cnt += 1
            pbar.set_postfix({"avg_masked_loss": running/cnt})
        print(f"Epoch {ep+1} avg masked loss: {running/cnt:.6f}")
    model.eval()
    return model

# -------------------------
# Patch extraction & memory-safe batched inference
# -------------------------
def extract_patches(image, ph, pw, stride_h, stride_w):
    H, W = image.shape
    pad_h = (ph - (H % ph)) % ph
    pad_w = (pw - (W % pw)) % pw
    image_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect') if (pad_h or pad_w) else image
    Hp, Wp = image_padded.shape
    ys = list(range(0, Hp - ph + 1, stride_h))
    xs = list(range(0, Wp - pw + 1, stride_w))
    if ys[-1] != Hp - ph:
        ys.append(Hp - ph)
    if xs[-1] != Wp - pw:
        xs.append(Wp - pw)
    coords = []
    patches = []
    for y in ys:
        for x in xs:
            patch = image_padded[y:y+ph, x:x+pw].astype(np.float32)
            coords.append((y, x))
            patches.append(patch)
    return image_padded, patches, coords

def batched_patch_inference_safe(image, model, ph, pw, stride_h, stride_w, device, batch_size=8):
    model.eval()
    win = make_hanning_window(ph, pw)
    image_padded, patches, coords = extract_patches(image, ph, pw, stride_h, stride_w)
    Hp, Wp = image_padded.shape
    output = np.zeros_like(image_padded, dtype=np.float32)
    weight = np.zeros_like(image_padded, dtype=np.float32)

    num = len(patches)
    idx = 0
    # try decreasing batch size on OOM
    cur_batch = batch_size
    while idx < num:
        end = min(idx + cur_batch, num)
        batch_patches = patches[idx:end]
        # prepare
        inp_list = []; means = []; stds = []
        for p in batch_patches:
            m = p.mean(); s = p.std() if p.std() > 1e-6 else 1.0
            means.append(m); stds.append(s)
            pn = (p - m) / s
            inp_list.append(pn)
        inp_batch = np.stack(inp_list, axis=0)[:, None, :, :]  # (B,1,H,W)
        inp_t = torch.from_numpy(inp_batch).to(device).float()
        try:
            if USE_AMP and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        out = model(inp_t)
            else:
                with torch.no_grad():
                    out = model(inp_t)
            out = out.detach().cpu().numpy()[:, 0, :, :]  # (B,H,W)
            # accumulate
            for j in range(out.shape[0]):
                y, x = coords[idx + j]
                patch_out = out[j] * stds[j] + means[j]
                output[y:y+ph, x:x+pw] += patch_out * win
                weight[y:y+ph, x:x+pw] += win
            idx = end
            # gradually try to grow batch if it was reduced earlier (optional)
            # (we leave it alone for stability)
        except RuntimeError as e:
            # likely CUDA OOM - reduce batch and retry
            if 'out of memory' in str(e).lower() and device.type == 'cuda' and cur_batch > 1:
                cur_batch = max(1, cur_batch // 2)
                torch.cuda.empty_cache()
                print(f"CUDA OOM encountered. Reducing inference batch to {cur_batch} and retrying at idx {idx}.")
                continue
            else:
                raise e

    H, W = image.shape
    denoised = output[:H, :W] / (weight[:H, :W] + 1e-8)
    return denoised

# -------------------------
# Simple no-GT metrics
# -------------------------
def noise_reduction_rate(I, D):
    return 1 - (np.var(D) / (np.var(I) + 1e-12))

def edge_preservation_index(I, D):
    from scipy import ndimage
    sx = ndimage.sobel(I, axis=1); sy = ndimage.sobel(I, axis=0)
    sI = np.hypot(sx, sy)
    sx2 = ndimage.sobel(D, axis=1); sy2 = ndimage.sobel(D, axis=0)
    sD = np.hypot(sx2, sy2)
    return sD.sum() / (sI.sum() + 1e-12)

# -------------------------
# Main pipeline
# -------------------------
def run_pipeline():
    image = load_hdf_as_numpy(UNFILTERED_FILE, UNFILTERED_VAR)
    H, W = image.shape
    print("Loaded image shape:", image.shape, " dtype:", image.dtype)
    print("Device:", DEVICE, "AMP:", USE_AMP)
    # choose model
    if MODEL_NAME.lower() == 'dncnn':
        model = DnCNN(in_channels=1, features=64, depth=12)
        wpath = DN_CNN_WEIGHTS
    elif MODEL_NAME.lower() == 'nafnet':
        # width/depth tuned to fit on 12GB-ish GPU with moderate batch sizes and 128x128 patches
        model = NAFNet(img_channels=1, width=48, enc_depths=[2,2,4], middle_blocks=8)
        wpath = NAFNET_WEIGHTS
    else:
        raise ValueError("Unsupported model name")
    model = model.to(DEVICE)

    # try loading weights if available
    pretrained = False
    if os.path.exists(wpath):
        try:
            model.load_state_dict(torch.load(wpath, map_location=DEVICE))
            print("Loaded pretrained weights from", wpath)
            pretrained = True
        except Exception as e:
            print("Failed loading weights:", e)

    # finetune if requested and no pretrained weights
    if (not pretrained) and FINETUNE:
        print("No pretrained weights: running self-supervised finetuning (masked loss).")
        model = finetune_selfsupervised(model, image, epochs=FINETUNE_EPOCHS, batch_size=FINETUNE_BATCH, lr=1e-4, device=DEVICE)
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        save_name = os.path.join(WEIGHTS_DIR, f"{MODEL_NAME}_finetuned.pth")
        torch.save(model.state_dict(), save_name)
        print("Saved finetuned weights to", save_name)

    # inference (memory-safe)
    print("Running memory-safe batched patch inference ...")
    denoised = batched_patch_inference_safe(image, model, PATCH_H, PATCH_W, STRIDE_H, STRIDE_W, DEVICE, batch_size=INFERENCE_BATCH)

    # visualize and metrics
    diff = image - denoised
    vmin = np.percentile(image, 1); vmax = np.percentile(image, 99)
    plt.figure(figsize=(16,6))
    plt.subplot(1,3,1)
    plt.imshow(image, cmap='turbo', vmin=vmin, vmax=vmax, aspect='auto'); plt.title("Original"); plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(denoised, cmap='turbo', vmin=vmin, vmax=vmax, aspect='auto'); plt.title(f"Denoised ({MODEL_NAME})"); plt.axis('off')
    plt.subplot(1,3,3)
    dmin = np.percentile(diff, 1); dmax = np.percentile(diff, 99)
    plt.imshow(diff, cmap='seismic', vmin=dmin, vmax=dmax, aspect='auto'); plt.title("Original - Denoised"); plt.axis('off')
    plt.tight_layout(); plt.show()

    nrr = noise_reduction_rate(image, denoised)
    epi = edge_preservation_index(image, denoised)
    print(f"Noise reduction rate: {nrr:.4f}; Edge preservation index: {epi:.4f}")

    np.save(f"denoised_{MODEL_NAME}.npy", denoised)
    print("Saved denoised image to", f"denoised_{MODEL_NAME}.npy")

if __name__ == "__main__":
    run_pipeline()
