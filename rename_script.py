# This script renames the data to how it should be
# return format:
# .jpg
# _aug_0.jpg
# _aug_1.jpg
# _noisy.jpg
# _aug_0_noisy.jpg
# _aug_1_noisy.jpg
  

from pathlib import Path
import re

img_data = Path("FF_Img_Data")

pat = re.compile(r"""
    ^(.*?)
    (?:\.jpg)?
    (?:_aug_(\d+))?
    (?:_noisy)*
    \.jpg$       
""", re.VERBOSE)

for frame_dir in img_data.iterdir():
    for p in frame_dir.glob("*.jpg"):
        m = pat.match(p.name)
        if not m:
            print(f"SKIP {p.name}")
            continue

        base, aug_idx = m.group(1), m.group(2)
        base = re.sub(r'(?:_noisy)+$', '', base)
        aug_part = f"_aug_{aug_idx}" if aug_idx is not None else ""
        noisy_part = "_noisy" if "_noisy" in p.name else ""
        
        new_name = f"{base}{aug_part}{noisy_part}.jpg"
        new_path = p.with_name(new_name)

        if new_path.exists():
            print(f"Already exists {new_path.name}")
            continue

        print(f"{p.name} -> {new_path.name}") 
        p.rename(new_path)