from pathlib import Path
from collections import defaultdict


def scan_tifs(root):
    folders = defaultdict(list)
    for p in Path(root).rglob('*.tif'):
        folders[p.parent].append(p)

    results = []
    for folder, files in folders.items():
        if len(files) == 1:
            results.append(files[0])  # 3D file
        else:
            results.append(folder)  # 2D sequence folder
    return results


# Usage:
for item in scan_tifs('/media/ghc/Ghc_data3/BRC'):
    print(item)