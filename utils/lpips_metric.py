# lpips_eval.py
# Usage:
#   pip install lpips pillow
#   python lpips_eval.py \
#     --orig_dir /path/to/originals \
#     --black_dir /path/to/blackbox \
#     --white_dir /path/to/whitebox \
#     --backbone alex \
#     --batch_size 16

import argparse, os, glob
from PIL import Image
import torch
import torch.nn.functional as F
import lpips  # https://github.com/richzhang/PerceptualSimilarity
import csv

def load_image(path, target_size=None):
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.BICUBIC)
    x = torch.from_numpy(np.array(img)).permute(2,0,1).float()  # [C,H,W], 0..255
    x = x / 255.0 * 2.0 - 1.0  # normalize to [-1, 1] for LPIPS
    return x

def build_pairs(orig_dir, other_dir):
    # Match by filename (ignores subfolders). Only include files present in both.
    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.webp")
    orig_files = []
    for e in exts:
        orig_files += glob.glob(os.path.join(orig_dir, "**", e), recursive=True)
    name_to_path = {os.path.basename(p): p for p in orig_files}
    pairs = []
    for e in exts:
        for p in glob.glob(os.path.join(other_dir, "**", e), recursive=True):
            name = os.path.basename(p)
            if name in name_to_path:
                pairs.append((name_to_path[name], p, name))
    return sorted(pairs, key=lambda t: t[2])  # sort by filename

def pad_or_resize_batch(batch, size_mode="resize", target_hw=None):
    # batch: list of tensors [C,H,W] in [-1,1]
    if size_mode == "resize":
        # resize all to the first image's size or given target_hw
        if target_hw is None:
            H, W = batch[0].shape[-2:]
        else:
            H, W = target_hw
        out = [F.interpolate(x.unsqueeze(0), size=(H,W), mode="bilinear", align_corners=False).squeeze(0) for x in batch]
        return torch.stack(out, 0)
    else:
        # simple center-crop to min size across batch (keeps aspect)
        H = min(x.shape[-2] for x in batch)
        W = min(x.shape[-1] for x in batch)
        out = []
        for x in batch:
            h, w = x.shape[-2:]
            top = (h - H)//2
            left = (w - W)//2
            out.append(x[:, top:top+H, left:left+W])
        return torch.stack(out, 0)

import numpy as np

@torch.inference_mode()
def eval_lpips(pairs, net="alex", batch_size=16, resize_mode="resize", out_csv=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net=net).to(device).eval()

    # Determine a canonical size from the first pair (only used if resize_mode='resize')
    if len(pairs) == 0:
        print("No matching pairs found.")
        return []
    # Pre-read first to set size
    first_o = Image.open(pairs[0][0]).convert("RGB")
    target_hw = first_o.size[1], first_o.size[0]  # PIL size is (W,H)

    results = []
    buf_o, buf_r, names = [], [], []
    def flush():
        if not names:
            return
        # stack and compute
        Xo = pad_or_resize_batch(buf_o, size_mode=resize_mode, target_hw=target_hw).to(device)
        Xr = pad_or_resize_batch(buf_r, size_mode=resize_mode, target_hw=target_hw).to(device)
        # LPIPS expects NCHW in [-1,1]
        d = loss_fn(Xo, Xr).squeeze().detach().cpu().numpy()  # shape [N]
        for n, val in zip(names, d.tolist()):
            results.append((n, float(val)))
        buf_o.clear(); buf_r.clear(); names.clear()

    for op, rp, name in pairs:
        o = Image.open(op).convert("RGB")
        r = Image.open(rp).convert("RGB")
        if resize_mode == "resize":
            # match to canonical target_hw based on first original
            o = o.resize((target_hw[1], target_hw[0]), Image.BICUBIC)
            r = r.resize((target_hw[1], target_hw[0]), Image.BICUBIC)
        to = torch.from_numpy(np.array(o)).permute(2,0,1).float()/255.0*2-1
        tr = torch.from_numpy(np.array(r)).permute(2,0,1).float()/255.0*2-1
        buf_o.append(to); buf_r.append(tr); names.append(name)
        if len(names) >= batch_size:
            flush()
    flush()

    if out_csv is not None:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", f"lpips_{net}"])
            writer.writerows(results)
        print(f"Saved: {out_csv}")
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", default="./attack_pic/Origin_pic")
    ap.add_argument("--black_dir", default='./attack_pic/black_attack/DCT_160Channel')#DCT_64Channel, DCT_160Channel, DCT_128Channel
    ap.add_argument("--white_dir", default='./attack_pic/white_attack_2/160channel')
    ap.add_argument("--backbone", choices=["alex","vgg","squeeze"], default="squeeze")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--resize_mode", choices=["resize","crop"], default="resize",
                    help="Resize all to a canonical size (default) or center-crop to min size in batch.")
    ap.add_argument("--out_dir", default="./lpips_160Channel_results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Black-box
    if args.black_dir:
        bb_pairs = build_pairs(args.orig_dir, args.black_dir)
        bb_csv = os.path.join(args.out_dir, f"lpips_black_{args.backbone}.csv")
        bb_res = eval_lpips(bb_pairs, net=args.backbone, batch_size=args.batch_size,
                            resize_mode=args.resize_mode, out_csv=bb_csv)
        print(f"[Black-box] N={len(bb_res)}  mean={np.mean([v for _,v in bb_res]):.4f}  std={np.std([v for _,v in bb_res]):.4f}")

    # White-box
    if args.white_dir:
        wb_pairs = build_pairs(args.orig_dir, args.white_dir)
        wb_csv = os.path.join(args.out_dir, f"lpips_white_{args.backbone}.csv")
        wb_res = eval_lpips(wb_pairs, net=args.backbone, batch_size=args.batch_size,
                            resize_mode=args.resize_mode, out_csv=wb_csv)
        print(f"[White-box] N={len(wb_res)}  mean={np.mean([v for _,v in wb_res]):.4f}  std={np.std([v for _,v in wb_res]):.4f}")

if __name__ == "__main__":
    main()
