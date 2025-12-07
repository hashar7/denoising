"""
advanced_denoise_pipeline.py

Integrated pipeline: DnCNN, TinyNAFNet, Full NAFNet (paper-like) with GPU support (RTX 3070 Ti).
- Loads HDF swath (your var), self-supervised finetune if no weights, patch-based batched inference on GPU,
- Mixed precision training/inference for speed & memory efficiency,
- Visualization & no-ground-truth evaluation measures.

Usage:
  - Place this file with your HDF file.
  - Optionally put pretrained weights in ./weights/dncnn.pth, ./weights/tinynaf.pth, ./weights/nafnet.pth
  - Run: python advanced_denoise_pipeline.py
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import windows
from tqdm import tqdm
from skimage import exposure

# -------------------------
# User / runtime parameters
# -------------------------
UNFILTERED_FILE = 'metm24_TA_251128_0101_9074_01_02A.hdf'
UNFILTERED_VAR = "m_m2413_53_80V"

WEIGHTS_DIR = "./weights"
DN_CNN_WEIGHTS = os.path.join(WEIGHTS_DIR, "dncnn.pth")
TINY_NAF_WEIGHTS = os.path.join(WEIGHTS_DIR, "tinynaf.pth")
NAFNET_WEIGHTS = os.path.join(WEIGHTS_DIR, "nafnet.pth")

# Choose model: 'dncnn', 'tinynaf', 'nafnet'
MODEL_NAME = 'nafnet'

# Patch & inference config (tunable)
PATCH_H = 256
PATCH_W = 256
STRIDE_H = PATCH_H // 2
STRIDE_W = PATCH_W // 2
BATCH_INFERENCE = 16   # number of patches processed at once during inference (GPU batched)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
PIN_MEMORY = True if DEVICE.type == "cuda" else False
torch.set_num_threads(4)
# Finetune config (self-supervised) - small on GPU
FINETUNE = True        # set False if you have supervised weights
FINETUNE_EPOCHS = 8
FINETUNE_BATCH = 16
LEARNING_RATE = 1e-4

# Mixed-precision
USE_AMP = True if DEVICE.type == "cuda" else False

# -------------------------
# Utilities
# -------------------------
def load_hdf_as_numpy(file_path, var_name):
    f = SD(file_path, SDC.READ)
    data = f.select(var_name).get()
    arr = np.array(data, dtype=np.float32)
    return arr

def make_hanning_window(h, w):
    win_h = windows.hann(h, sym=False)
    win_w = windows.hann(w, sym=False)
    win2d = np.outer(win_h, win_w).astype(np.float32)
    win2d += 1e-6
    return win2d

# -------------------------
# Models
# -------------------------
# 1) DnCNN (small)
class DnCNN(nn.Module):
    def __init__(self, in_channels=1, features=64, depth=12):
        super().__init__()
        layers = [nn.Conv2d(in_channels, features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(features, features, 3, padding=1, bias=False),
                       nn.BatchNorm2d(features),
                       nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(features, in_channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        noise = self.net(x)
        return x - noise

# 2) Tiny NAF-style (light)
class SimpleNAFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.act = nn.GELU()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.act(y)
        y = self.conv3(y)
        return x + self.beta * y

class TinyNAFNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, width=32, num_blocks=6):
        super().__init__()
        self.entry = nn.Conv2d(in_ch, width, 3, padding=1)
        self.body = nn.Sequential(*[SimpleNAFBlock(width) for _ in range(num_blocks)])
        self.exit = nn.Conv2d(width, out_ch, 3, padding=1)
    def forward(self, x):
        fe = self.entry(x)
        fb = self.body(fe)
        out = self.exit(fb)
        return x - out

# 3) Full paper-like NAFNet (encoder-decoder U-Net style)
# Implement core building blocks inspired by the NAFNet paper:
class SimpleGate(nn.Module):
    def forward(self, x):
        # split channel dim in half and multiply
        a, b = x.chunk(2, dim=1)
        return a * b

class SimpleChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1, bias=True)
    def forward(self, x):
        v = self.avg_pool(x)
        v = self.conv(v)
        return x * torch.sigmoid(v)

class NAFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.LayerNorm([channels, 1, 1], elementwise_affine=True) if False else nn.Identity()
        self.pw1 = nn.Conv2d(channels, channels*2, 1, bias=True)
        self.conv = nn.Conv2d(channels*2, channels*2, 3, padding=1, groups=1)
        self.sg = SimpleGate()
        self.pw2 = nn.Conv2d(channels, channels, 1, bias=True)
        self.sca = SimpleChannelAttention(channels)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
    def forward(self, x):
        res = x
        x = self.pw1(x)
        x = self.conv(x)
        x = self.sg(x)   # halves and multiplies -> channels restored to channels
        x = self.pw2(x)
        x = self.sca(x)
        x = res + self.beta * x
        # simple FFN
        y = x
        y = nn.Conv2d(x.shape[1], x.shape[1]*2, 1).to(x.device)(y)  # not ideal to create in forward; we'll replace below
        return x  # Note: For simplicity & memory, use residual only

# Implement a proper NAFBlock without creating conv in forward:
class NAFBlockV2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pw1 = nn.Conv2d(channels, channels*2, 1, bias=True)
        self.dw  = nn.Conv2d(channels*2, channels*2, 3, padding=1, groups=channels*2)
        self.sg  = SimpleGate()
        self.pw2 = nn.Conv2d(channels, channels, 1, bias=True)
        self.sca = SimpleChannelAttention(channels)
        # simple FFN
        self.ffn1 = nn.Conv2d(channels, channels*2, 1, bias=True)
        self.ffn_dw = nn.Conv2d(channels*2, channels*2, 3, padding=1, groups=channels*2)
        self.ffn2 = nn.Conv2d(channels, channels, 1, bias=True)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.act = nn.GELU()
    def forward(self, x):
        identity = x
        x = self.pw1(x)
        x = self.dw(x)
        x = self.sg(x)
        x = self.act(x)
        # reduce channels back
        x = x.reshape(x.size(0), -1, x.size(2), x.size(3))
        x = self.pw2(x)
        x = self.sca(x)
        x = identity + self.beta * x

        # FFN
        y = self.ffn1(x)
        y = self.ffn_dw(y)
        y = self.sg(y)
        y = self.act(y)
        y = self.ffn2(y)
        x = x + self.gamma * y
        return x

class NAFNet(nn.Module):
    def __init__(self, img_channels=1, width=48, enc_depths=[2,2,4,8], middle_blocks=8):
        super().__init__()
        self.entry = nn.Conv2d(img_channels, width, 3, padding=1)
        # encoder
        self.encs = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for d in enc_depths:
            blocks = nn.Sequential(*[NAFBlockV2(ch) for _ in range(d)])
            self.encs.append(blocks)
            self.downs.append(nn.Conv2d(ch, ch*2, 2, stride=2))  # downsample
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
            # crop or pad if mismatch
            if x.shape != feat.shape:
                # center-crop or pad
                min_h = min(x.shape[2], feat.shape[2])
                min_w = min(x.shape[3], feat.shape[3])
                x = x[:, :, :min_h, :min_w]
                feat = feat[:, :, :min_h, :min_w]
            x = x + feat
            x = dec(x)
        out = self.exit(x)
        return x - out   # residual subtract

# -------------------------
# Dataset for self-supervised finetuning
# -------------------------
class ImagePatchDataset(Dataset):
    def __init__(self, image, patch_h=128, patch_w=128, num_samples=4000, mask_ratio=0.05):
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
        # create mask
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
        inp_t = torch.from_numpy(inp[None]).float()   # (1,H,W)
        tgt_t = torch.from_numpy(tgt[None]).float()
        mask_t = torch.from_numpy(mask[None]).float()
        return inp_t, tgt_t, mask_t

# -------------------------
# Finetune function (self-supervised using masked loss)
# -------------------------
def finetune_selfsupervised(model, image, epochs=5, batch_size=16, lr=1e-4, device=DEVICE):
    ds = ImagePatchDataset(image, patch_h=128, patch_w=128, num_samples=3000, mask_ratio=0.05)
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
                out = model(inp)
                l = loss_fn(out, tgt)
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
# Efficient batched patch inference on GPU (or CPU)
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
            coords.append((y,x))
            patches.append(patch)
    return image_padded, patches, coords

def batched_patch_inference(image, model, ph, pw, stride_h, stride_w, device, batch_size=BATCH_INFERENCE):
    model.eval()
    win = make_hanning_window(ph, pw)
    image_padded, patches, coords = extract_patches(image, ph, pw, stride_h, stride_w)
    Hp, Wp = image_padded.shape
    output = np.zeros_like(image_padded, dtype=np.float32)
    weight = np.zeros_like(image_padded, dtype=np.float32)
    # process in batches
    num = len(patches)
    idx = 0
    with torch.no_grad():
        while idx < num:
            batch_patches = patches[idx: idx + batch_size]
            # prepare tensor: normalize per-patch and stack
            inp_list = []
            means = []
            stds = []
            for p in batch_patches:
                m = p.mean(); s = p.std() if p.std() > 1e-6 else 1.0
                means.append(m); stds.append(s)
                pn = (p - m) / s
                inp_list.append(pn)
            inp_batch = np.stack(inp_list, axis=0)[:, None, :, :]  # (B,1,H,W)
            inp_t = torch.from_numpy(inp_batch).to(device).float()
            if USE_AMP and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    out = model(inp_t)
            else:
                out = model(inp_t)
            out = out.detach().cpu().numpy()[:, 0, :, :]  # (B,H,W)
            # denormalize and accumulate
            for j in range(out.shape[0]):
                y, x = coords[idx + j]
                patch_out = out[j] * stds[j] + means[j]
                output[y:y+ph, x:x+pw] += patch_out * win
                weight[y:y+ph, x:x+pw] += win
            idx += batch_size
    # crop and normalize
    H, W = image.shape
    denoised = output[:H, :W] / (weight[:H, :W] + 1e-8)
    return denoised

# -------------------------
# No-ground-truth evaluation helpers
# -------------------------
def noise_reduction_rate(I, D):
    return 1 - (np.var(D) / (np.var(I) + 1e-12))

def edge_preservation_index(I, D):
    from scipy import ndimage
    sx = ndimage.sobel(I, axis=1)
    sy = ndimage.sobel(I, axis=0)
    sI = np.hypot(sx, sy)
    sx2 = ndimage.sobel(D, axis=1)
    sy2 = ndimage.sobel(D, axis=0)
    sD = np.hypot(sx2, sy2)
    return sD.sum() / (sI.sum() + 1e-12)

# -------------------------
# Main runner
# -------------------------
def run_pipeline():
    # 1) load image
    image = load_hdf_as_numpy(UNFILTERED_FILE, UNFILTERED_VAR)
    H, W = image.shape
    print("Loaded image:", image.shape, " dtype:", image.dtype)
    print("Using device:", DEVICE, "AMP enabled:", USE_AMP)

    # 2) choose model & try loading weights
    if MODEL_NAME.lower() == 'dncnn':
        model = DnCNN(in_channels=1, features=64, depth=12)
        wpath = DN_CNN_WEIGHTS
    elif MODEL_NAME.lower() == 'tinynaf':
        model = TinyNAFNet(in_ch=1, out_ch=1, width=32, num_blocks=8)
        wpath = TINY_NAF_WEIGHTS
    elif MODEL_NAME.lower() == 'nafnet':
        # a moderately-sized NAFNet for 12GB GPU
        model = NAFNet(img_channels=1, width=48, enc_depths=[2,2,4], middle_blocks=8)
        wpath = NAFNET_WEIGHTS
    else:
        raise ValueError("Unknown model")
    model = model.to(DEVICE)

    pretrained = False
    if os.path.exists(wpath):
        try:
            model.load_state_dict(torch.load(wpath, map_location=DEVICE))
            print("Loaded pretrained weights from", wpath)
            pretrained = True
        except Exception as e:
            print("Failed loading weights:", e)

    # 3) finetune self-supervised if no pretrained
    if (not pretrained) and FINETUNE:
        print("No pretrained weights found — running self-supervised finetuning")
        model = finetune_selfsupervised(model, image, epochs=FINETUNE_EPOCHS, batch_size=FINETUNE_BATCH, lr=LEARNING_RATE, device=DEVICE)
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        save_path = os.path.join(WEIGHTS_DIR, f"{MODEL_NAME}_finetuned.pth")
        torch.save(model.state_dict(), save_path)
        print("Saved finetuned weights to", save_path)

    # 4) batched patch inference on GPU/CPU
    print("Running patch-based batched inference ...")
    denoised = batched_patch_inference(image, model, PATCH_H, PATCH_W, STRIDE_H, STRIDE_W, DEVICE, batch_size=BATCH_INFERENCE)

    # 5) visualize
    diff = image - denoised
    vmin = np.percentile(image, 1)
    vmax = np.percentile(image, 99)

    plt.figure(figsize=(18,6))
    plt.subplot(1,3,1)
    plt.imshow(image, cmap='turbo', vmin=vmin, vmax=vmax, aspect='auto')
    plt.title("Original (Unfiltered)")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(denoised, cmap='turbo', vmin=vmin, vmax=vmax, aspect='auto')
    plt.title(f"Denoised ({MODEL_NAME})")
    plt.axis('off')

    plt.subplot(1,3,3)
    dmin = np.percentile(diff, 1); dmax = np.percentile(diff, 99)
    plt.imshow(diff, cmap='seismic', vmin=dmin, vmax=dmax, aspect='auto')
    plt.title("Difference (Original - Denoised)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 6) metrics (no ground truth)
    nrr = noise_reduction_rate(image, denoised)
    epi = edge_preservation_index(image, denoised)
    print(f"Noise reduction rate: {nrr:.4f}")
    print(f"Edge preservation index: {epi:.4f}")

    # 7) save output
    outname = f"denoised_{MODEL_NAME}.npy"
    np.save(outname, denoised)
    print("Saved denoised image to", outname)