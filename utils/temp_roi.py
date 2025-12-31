import glob
import tifffile as tiff
import numpy as np
from tqdm import tqdm

root = '/media/ghc/Ghc_data3/BRC/aisr/THX10SDM20xw/raw/'
xlist = sorted(glob.glob(root + '*.tif'))
x3d = []
for x in tqdm(xlist[::4]):
    print(x)
    img = tiff.imread(x)
    print(img.shape)
    x3d.append(img[4096:4096*2, 4096:4096*2])

x3d = np.stack(x3d, axis=0)
tiff.imwrite(root + 'roi.tif', x3d)