from pathlib import Path
import re, random
from collections import defaultdict

# ---------- CONSTANTS --------------------------------------------------
ROOT = Path("FF_Img_Data")       # dataset root (contains original/, faceswap/ …)
OUT  = Path("train_test_data")   # output folder
IMG_GLOB = "*.jpg"                # pattern for frame files
SEED = 42                         # RNG seed for reproducibility
# -----------------------------------------------------------------------

random.seed(SEED)
OUT.mkdir(exist_ok=True)

# 1) Gather frames ------------------------------------------------------
vid2paths: dict[str, list[Path]] = defaultdict(list)
fake_videos: set[str] = set()

for cls in ROOT.iterdir():
    if not cls.is_dir():
        continue
    is_fake_cls = cls.name != "original"
    for img in cls.glob(IMG_GLOB):
        m = re.match(r"(.+?)_frame_", img.name)
        if not m:
            continue  # skip files that don't match <video>_frame_
        vid = m.group(1)
        vid2paths[vid].append(img)
        if is_fake_cls:
            fake_videos.add(vid)

real_videos = [v for v in vid2paths if v not in fake_videos]
fake_videos = list(fake_videos)

if not real_videos or not fake_videos:
    raise SystemExit("Dataset must contain both real and fake videos.")

# 2) Balance by VIDEO count --------------------------------------------
num_each = min(len(real_videos), len(fake_videos))
random.shuffle(real_videos)
random.shuffle(fake_videos)
kept_videos = real_videos[:num_each] + fake_videos[:num_each]
random.shuffle(kept_videos)

# 3) Split videos 80 / 10 / 10 -----------------------------------------
num_vids = len(kept_videos)
train_cut = int(num_vids * 0.8)
val_cut   = int(num_vids * 0.9)

splits = {
    "train": kept_videos[:train_cut],
    "val"  : kept_videos[train_cut:val_cut],
    "test" : kept_videos[val_cut:],
}

# 4) For every split, balance by FRAME count & write list ---------------
print("\nFrame list files written under", OUT)
for split, vids in splits.items():
    real_lines, fake_lines = [], []
    for vid in vids:
        lines = [str(p.relative_to(ROOT.parent)) for p in vid2paths[vid]]
        if vid in fake_videos:
            fake_lines.extend(lines)
        else:
            real_lines.extend(lines)

    # Down‑sample majority frames to match minority
    n = min(len(real_lines), len(fake_lines))
    random.shuffle(real_lines)
    random.shuffle(fake_lines)
    real_lines, fake_lines = real_lines[:n], fake_lines[:n]

    combined = real_lines + fake_lines
    random.shuffle(combined)
    (OUT / f"{split}.txt").write_text("\n".join(combined))

    print(f"{split:<5}: {len(vids):4} videos | {len(combined):6} frames written")

print("\nFrame count per split (real | fake | %fake)")
for split, vids in splits.items():
    real_cnt = fake_cnt = 0
    for vid in vids:
        if vid in fake_videos:
            fake_cnt += len(vid2paths[vid])
        else:
            real_cnt += len(vid2paths[vid])
    # After down‑sampling, counts are equal to the *written* n each
    n_written = min(real_cnt, fake_cnt) * 2
    pct = 50.0  # by construction
    print(f"{split:<5} {n_written//2:6} | {n_written//2:6} | {pct:5.2f}% (total {n_written})")
