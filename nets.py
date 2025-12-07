# -------------------------
# Patching & batched inference (memory-safe)
# -------------------------
def extract_patches(image, ph, pw, stride_h, stride_w):
    H, W = image.shape
    pad_h = (ph - (H % ph)) % ph
    pad_w = (pw - (W % pw)) % pw
    image_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect') if (pad_h or pad_w) else image.copy()
    Hp, Wp = image_padded.shape
    ys = list(range(0, Hp - ph + 1, stride_h))
    xs = list(range(0, Wp - pw + 1, stride_w))
    if ys[-1] != Hp - ph:
        ys.append(Hp - ph)
    if xs[-1] != Wp - pw:
        xs.append(Wp - pw)
    coords = []
    for y in ys:
        for x in xs:
            coords.append((y, x))
    return image_padded, coords

def batched_patch_inference(image, model, ph, pw, stride_h, stride_w, device, batch_size=2):
    model.eval()
    win = make_hanning_window(ph, pw)
    image_padded, coords = extract_patches(image, ph, pw, stride_h, stride_w)
    Hp, Wp = image_padded.shape
    output = np.zeros_like(image_padded, dtype=np.float32)
    weight = np.zeros_like(image_padded, dtype=np.float32)

    num = len(coords)
    idx = 0
    with torch.no_grad():
        while idx < num:
            bs = min(batch_size, num - idx)
            inp_list = []
            means = []
            stds = []
            for j in range(bs):
                y, x = coords[idx + j]
                patch = image_padded[y:y+ph, x:x+pw].astype(np.float32)
                m = patch.mean(); s = patch.std() if patch.std() > 1e-6 else 1.0
                pn = (patch - m) / s
                inp_list.append(pn)
                means.append(m); stds.append(s)
            inp_batch = np.stack(inp_list, axis=0)[:, None, :, :]  # (B,1,H,W)
            inp_t = torch.from_numpy(inp_batch).to(device).float()
            if USE_AMP and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    out = model(inp_t)
            else:
                out = model(inp_t)
            out_np = out.detach().cpu().numpy()[:, 0, :, :]  # (B,H,W)
            # accumulate
            for j in range(bs):
                y, x = coords[idx + j]
                patch_out = out_np[j] * stds[j] + means[j]
                output[y:y+ph, x:x+pw] += patch_out * win
                weight[y:y+ph, x:x+pw] += win
            idx += bs

    H, W = image.shape
    denoised = output[:H, :W] / (weight[:H, :W] + 1e-8)
    return denoised

# -------------------------
# No-GT evaluation helpers
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