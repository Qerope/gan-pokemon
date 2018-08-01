import glob

import numpy as np
from PIL import Image


def load_sprites(folder='data', size=64):
    imgs = []
    for path in glob.glob(folder + '/*.png'):
        im = Image.open(path).convert('RGBA').resize((size, size))
        imgs.append(np.asarray(im.convert('RGB'), dtype=np.float32))
    return np.stack(imgs) / 127.5 - 1.0
