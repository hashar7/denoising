"""
denoise_with_nafnet.py

Adds a full paper-like NAFNet to the previous denoising pipeline and integrates it with DnCNN and TinyNAFNet.

Usage:
- Configure user parameters at the top (MODEL_NAME, weight paths, HDF path)
- Run: python denoise_with_nafnet.py
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

# -----------------------
# USER PARAMETERS
# -----------------------
unfiltered_file_name = 'metm24_TA_251128_0101_9074_01_02A.hdf'
unfiltered_var_name = "m_m2413_53_80V"

# Choose model: 'dncnn', 'nafnet_tiny', 'nafnet_full'
MODEL_NAME = 'nafnet_full'

# Weights
WEIGHTS_DIR = "./weights"
DN_CNN_WEIGHTS = os.path.join(WEIGHTS_DIR, "dncnn.pth")
NAFNET_TINY_WEIGHTS = os.path.join(WEIGHTS_DIR, "nafnet.pth")
NAFNET_FULL_WEIGHTS = os.path.join(WEIGHTS_DIR, "nafnet_full.pth")

# Inference / patching config
PATCH_H = 128
PATCH_W = 256
STRIDE_H = PATCH_H // 2
STRIDE_W = PATCH_W // 2

# Finetuning/training options
FINETUNE = True
FINETUNE_EPOCHS = 6
FINETUNE_BATCH = 8
LEARNING_RATE = 1e-4

# Hardware / performance
FORCE_CPU = False      # set True to forbid GPU usage
NUM_THREADS = 4
torch.set_num_threads(NUM_THREADS)
np.random.seed(0)
torch.manual_seed(0)

# Device selection (auto GPU if available and not forced off)
DEVICE = torch.device("cuda" if torch.cuda.is_available() and (not FORCE_CPU) else "cpu")
print("Using device:", DEVICE)

# -----------------------
# HDF LOADER
# -----------------------
def load_hdf_as_numpy(file_path, var_name):
    f = SD(file_path, SDC.READ)
    data = f.select(var_name).get()
    arr = np.array(data, dtype=np.float32)
    return arr

# -----------------------
# DnCNN (unchanged / small)
# -----------------------
class DnCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_layers=12):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.net(x)
        return x - noise

# -----------------------
# Tiny NAFNet (as before)
# -----------------------
class SimpleNAFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.gelu = nn.GELU()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.gelu(y)
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

# -----------------------
# Full paper-like NAFNet implementation (compact, faithful)
#
# Based on "NAFNet: Nonlinear Activation Free Network for Image Restoration"
# Key components implemented here:
#  - SimpleGate (2-channel split + element-wise multiply)
#  - Local Attention via depthwise conv + pointwise conv
#  - LayerNorm (channel-last / channel-first variants)
#  - NAFBlock repeated in encoder/decoder style
#
# This is a compact readable implementation tuned for image restoration tasks.
# -----------------------

class LayerNormChannelWise(nn.Module):
    """LayerNorm over channels for 2D conv feature maps (N, C, H, W)"""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        # compute mean and var across channels for each pixel? The original NAF uses LN like layernorm across channels
        # We'll normalize per-channel (channel-wise LN) for simplicity and stability.
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.gamma + self.beta

class SimpleGate(nn.Module):
    def forward(self, x):
        # split channels in half and multiply
        c = x.shape[1]
        assert c % 2 == 0, "SimpleGate requires even number of channels"
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, channels, dw_expand=2, ffn_expand=2, dropout_rate=0.0):
        super().__init__()
        dw_chs = channels * dw_expand
        ffn_chs = channels * ffn_expand

        # Layernorm (channel-wise)
        self.norm1 = LayerNormChannelWise(channels)
        # Pointwise conv to expansion
        self.pw1 = nn.Conv2d(channels, dw_chs, kernel_size=1, bias=True)
        # Depthwise conv (local attention-ish)
        self.dwconv = nn.Conv2d(dw_chs, dw_chs, kernel_size=3, padding=1, groups=dw_chs, bias=True)
        # SimpleGate
        self.sg = SimpleGate()
        # Pointwise conv to channels
        self.pw2 = nn.Conv2d(dw_chs // 2, channels, kernel_size=1, bias=True)

        # FFN
        self.norm2 = LayerNormChannelWise(channels)
        self.pw3 = nn.Conv2d(channels, ffn_chs, kernel_size=1, bias=True)
        self.pw4 = nn.Conv2d(ffn_chs // 2, channels, kernel_size=1, bias=True)

        # learnable scaling
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        # main branch
        y = self.norm1(x)
        y = self.pw1(y)
        y = self.dwconv(y)
        y = self.sg(y)
        y = self.pw2(y)
        y = self.dropout(y)
        x = x + self.beta * y

        # ff branch
        z = self.norm2(x)
        z = self.pw3(z)
        z = self.sg(z)
        z = self.pw4(z)
        z = self.dropout(z)
        return x + self.gamma * z

class NAFNetFull(nn.Module):
    """
    A compact encoder-decoder style NAFNet.
    Stacks NAFBlocks in each stage with down/up sampling.
    """

    def __init__(self, img_channels=1, width=48, enc_blocks=[2, 2, 4, 8], dec_blocks=[8,4,2,2]):
        super().__init__()
        # entry projection
        self.entry = nn.Conv2d(img_channels, width, 3, padding=1)

        # encoder
        self.encs = nn.ModuleList()
        self.downs = nn.ModuleList()
        chs = width
        for n in enc_blocks:
            blocks = nn.Sequential(*[NAFBlock(chs) for _ in range(n)])
            self.encs.append(blocks)
            # downsample conv (stride 2)
            self.downs.append(nn.Conv2d(chs, chs * 2, kernel_size=2, stride=2))
            chs = chs * 2

        # bottleneck
        self.bottleneck = nn.Sequential(*[NAFBlock(chs) for _ in range(2)])

        # decoder
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for n in dec_blocks:
            # upsample conv transpose or nearest+conv
            self.ups.append(nn.ConvTranspose2d(chs, chs // 2, kernel_size=2, stride=2))
            chs = chs // 2
            self.decs.append(nn.Sequential(*[NAFBlock(chs) for _ in range(n)]))

        # exit projection
        self.exit = nn.Conv2d(width, img_channels, 3, padding=1)

    def forward(self, x):
        # store encoder intermediate for skip connections
        x = self.entry(x)
        enc_features = []
        for blocks, down in zip(self.encs, self.downs):
            x = blocks(x)
            enc_features.append(x)
            x = down(x)
        # bottleneck
        x = self.bottleneck(x)
        # decoder
        for up, dec, skip in zip(self.ups, self.decs, reversed(enc_features)):
            x = up(x)
            # if shapes mismatch due to odd sizes, center-crop skip
            if x.shape[-2:] != skip.shape[-2:]:
                # crop or pad skip to match
                sh, sw = skip.shape[-2:]
                x = F.interpolate(x, size=(sh, sw), mode='bilinear', align_corners=False)
            x = x + skip
            x = dec(x)
        out = self.exit(x)
        # predict residual and subtract
        return x - out  # NOTE: subtract output (residual) from original x; if unstable, change to return x - out

# -----------------------
# PATCH + HANNING WINDOW INFERENCE (same as before)
# -----------------------
def make_hanning_window(h, w):
    win_h = windows.hann(h, sym=False)
    win_w = windows.hann(w, sym=False)
    win2d = np.outer(win_h, win_w).astype(np.float32)
    win2d = win2d + 1e-6
    return win2d

def sliding_window_inference(image, model, patch_h, patch_w, stride_h, stride_w, device, use_amp=False):
    model.eval()
    H, W = image.shape
    pad_h = (patch_h - (H % patch_h)) % patch_h
    pad_w = (patch_w - (W % patch_w)) % patch_w
    if pad_h > 0 or pad_w > 0:
        image_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    else:
        image_padded = image
    Hp, Wp = image_padded.shape

    win = make_hanning_window(patch_h, patch_w)
    output = np.zeros_like(image_padded, dtype=np.float32)
    weight = np.zeros_like(image_padded, dtype=np.float32)

    ys = list(range(0, Hp - patch_h + 1, stride_h))
    xs = list(range(0, Wp - patch_w + 1, stride_w))
    if ys[-1] != Hp - patch_h:
        ys.append(Hp - patch_h)
    if xs[-1] != Wp - patch_w:
        xs.append(Wp - patch_w)

    # move model to eval device already outside
    scaler = torch.cuda.amp.autocast if use_amp else (lambda: (yield))  # hacky no-op if not using amp
    # We'll not use that generator hack; instead use context manager when use_amp True.
    for y in ys:
        for x in xs:
            patch = image_padded[y:y+patch_h, x:x+patch_w]
            mean = patch.mean()
            std = patch.std() if patch.std() > 1e-6 else 1.0
            patch_n = (patch - mean) / std
            inp = torch.from_numpy(patch_n[None, None, :, :]).to(device)
            with torch.no_grad():
                if use_amp and device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        out = model(inp).cpu().numpy()[0, 0]
                else:
                    out = model(inp).cpu().numpy()[0, 0]
            out = out * std + mean
            output[y:y+patch_h, x:x+patch_w] += out * win
            weight[y:y+patch_h, x:x+patch_w] += win

    output = output[:H, :W] / (weight[:H, :W] + 1e-8)
    return output

# -----------------------
# Self-supervised dataset (fixed shapes)
# -----------------------
class ImagePatchDataset(Dataset):
    def __init__(self, image, patch_h=64, patch_w=64, num_samples=2000, mask_ratio=0.05):
        self.image = image
        self.H, self.W = image.shape
        self.ph = patch_h
        self.pw = patch_w
        self.num_samples = num_samples
        self.mask_ratio = mask_ratio
        self.coords = []
        for _ in range(num_samples):
            y = np.random.randint(0, max(1, self.H - patch_h + 1))
            x = np.random.randint(0, max(1, self.W - patch_w + 1))
            self.coords.append((y, x))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        patch = self.image[y:y+self.ph, x:x+self.pw].astype(np.float32)
        mask = np.zeros_like(patch, dtype=np.float32)
        num_mask = int(self.mask_ratio * patch.size)
        coords = np.random.choice(patch.size, size=num_mask, replace=False)
        flat = mask.ravel()
        flat[coords] = 1.0
        mask = flat.reshape(patch.shape)
        masked_patch = patch.copy()
        local_mean = np.mean(patch)
        masked_patch[mask.astype(bool)] = local_mean
        mean = masked_patch.mean()
        std = masked_patch.std() if masked_patch.std() > 1e-6 else 1.0
        inp = (masked_patch - mean) / std
        target = (patch - mean) / std
        inp_t = torch.from_numpy(inp[None, :, :]).float()    # (1, H, W)
        target_t = torch.from_numpy(target[None, :, :]).float()
        mask_t = torch.from_numpy(mask[None, :, :]).float()
        return inp_t, target_t, mask_t

# -----------------------
# Self-supervised fine-tune (works on GPU if available)
# -----------------------
def finetune_selfsupervised(model, image, epochs=4, batch_size=8, lr=1e-4, device=torch.device("cpu"), use_amp=True):
    print("Starting self-supervised fine-tuning. Device:", device, "amp:", use_amp)
    ds = ImagePatchDataset(image, patch_h=64, patch_w=64, num_samples=1200)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=(device.type=='cuda'))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss(reduction="none")
    model.to(device)
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type=='cuda'))
    for ep in range(epochs):
        running = 0.0
        cnt = 0
        pbar = tqdm(dl, desc=f"Finetune ep{ep+1}/{epochs}")
        for inp, target, mask in pbar:
            inp = inp.to(device); target = target.to(device); mask = mask.to(device)
            opt.zero_grad()
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    out = model(inp)
                    l_pix = loss_fn(out, target)
                    masked_loss = (l_pix * mask).sum() / (mask.sum() + 1e-8)
                scaler.scale(masked_loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                out = model(inp)
                l_pix = loss_fn(out, target)
                masked_loss = (l_pix * mask).sum() / (mask.sum() + 1e-8)
                masked_loss.backward()
                opt.step()
            running += masked_loss.item()
            cnt += 1
            pbar.set_postfix({'loss': running / cnt})
        print(f"Epoch {ep+1} avg masked loss: {running/cnt:.6f}")
    model.eval()
    return model

# -----------------------
# Main pipeline
# -----------------------
def run_pipeline():
    # load
    image = load_hdf_as_numpy(unfiltered_file_name, unfiltered_var_name)
    H, W = image.shape
    print("Loaded image shape:", image.shape)

    # choose model
    if MODEL_NAME.lower() == 'dncnn':
        model = DnCNN(in_channels=1, out_channels=1, num_features=48, num_layers=12)
        weights_path = DN_CNN_WEIGHTS
    elif MODEL_NAME.lower() == 'nafnet_tiny':
        model = TinyNAFNet(in_ch=1, out_ch=1, width=32, num_blocks=8)
        weights_path = NAFNET_TINY_WEIGHTS
    elif MODEL_NAME.lower() == 'nafnet_full':
        # instantiate a reasonably sized full nafnet
        model = NAFNetFull(img_channels=1, width=48, enc_blocks=[2,2,4], dec_blocks=[4,2,2])
        weights_path = NAFNET_FULL_WEIGHTS
    else:
        raise ValueError("Unsupported model name")

    model.to(DEVICE)

    # attempt to load pretrained
    pretrained_available = False
    if os.path.exists(weights_path):
        print("Loading weights from", weights_path)
        sd = torch.load(weights_path, map_location=DEVICE)
        try:
            model.load_state_dict(sd)
            pretrained_available = True
        except Exception as e:
            print("Failed to load state_dict:", e)
            # try partial loading
            model_state = model.state_dict()
            loaded = {k:v for k,v in sd.items() if k in model_state and v.shape == model_state[k].shape}
            model_state.update(loaded)
            model.load_state_dict(model_state)
            print("Loaded partial weights for matching keys.")
            pretrained_available = True if len(loaded)>0 else False
    else:
        print("No pretrained weights found at", weights_path)

    # finetune if no weights (or optionally always finetune)
    if (not pretrained_available) and FINETUNE:
        model = finetune_selfsupervised(model, image, epochs=FINETUNE_EPOCHS, batch_size=FINETUNE_BATCH, lr=LEARNING_RATE, device=DEVICE, use_amp=True)
        # save finetuned
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        save_name = os.path.join(WEIGHTS_DIR, f"{MODEL_NAME}_finetuned.pth")
        torch.save(model.state_dict(), save_name)
        print("Saved finetuned model to", save_name)

    # run inference (use amp on GPU)
    use_amp = True if (DEVICE.type == 'cuda') else False
    denoised = sliding_window_inference(image, model, PATCH_H, PATCH_W, STRIDE_H, STRIDE_W, DEVICE, use_amp=use_amp)

    # visualize
    diff = image - denoised
    vmin = np.percentile(image, 2)
    vmax = np.percentile(image, 98)
    plt.figure(figsize=(18,6))
    plt.subplot(1,3,1); plt.imshow(image, cmap='turbo', vmin=vmin, vmax=vmax, aspect='auto'); plt.title("Original"); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(denoised, cmap='turbo', vmin=vmin, vmax=vmax, aspect='auto'); plt.title(f"Denoised ({MODEL_NAME})"); plt.axis('off')
    dmin = np.percentile(diff,1); dmax = np.percentile(diff,99)
    plt.subplot(1,3,3); plt.imshow(diff, cmap='seismic', vmin=dmin, vmax=dmax, aspect='auto'); plt.title("Original - Denoised"); plt.axis('off')
    plt.tight_layout(); plt.show()

    # save result
    out_path = f"denoised_{MODEL_NAME}.npy"
    np.save(out_path, denoised)
    print("Saved denoised image to", out_path)