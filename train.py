import argparse
import os

import numpy as np

import data
import models

OPT_ARGS = dict(lr=2e-4, beta_1=0.5)
from PIL import Image


def save_grid(gen, step, n=5):
    z = np.random.normal(size=(n * n, models.LATENT))
    imgs = ((gen.predict(z) + 1) * 127.5).astype(np.uint8)
    grid = imgs.reshape(n, n, 64, 64, 3).transpose(0, 2, 1, 3, 4).reshape(n * 64, n * 64, 3)
    os.makedirs('samples', exist_ok=True)
    Image.fromarray(grid).save('samples/%06d.png' % step)


def main():
    x = data.load_sprites()
    gen = models.build_generator()
    disc = models.build_discriminator()
    disc.compile(models.Adam(**models.OPT_ARGS), 'binary_crossentropy')
    gan = models.build_gan(gen, disc)
    gan.compile(models.Adam(**models.OPT_ARGS), 'binary_crossentropy')

    batch = 64
    step = 0
    for epoch in range(300):
        idx = np.random.permutation(len(x))
        for k in range(0, len(x) - batch + 1, batch):
            real = x[idx[k:k + batch]]
            z = np.random.normal(size=(batch, models.LATENT))
            fake = gen.predict(z)
            real_labels = np.where(np.random.rand(batch, 1) < 0.05, 0.0, 1.0)
            d_real = disc.train_on_batch(real, real_labels)
            d_fake = disc.train_on_batch(fake, np.zeros((batch, 1)))
            d_loss = d_real + d_fake
            g_loss = gan.train_on_batch(z, np.ones((batch, 1)))
            step += 1
            if step % 200 == 0:
                save_grid(gen, step)
                print(step, d_loss, g_loss)
            if step % 1000 == 0:
                os.makedirs('checkpoints', exist_ok=True)
                gen.save_weights('checkpoints/gen_%d.h5' % step)
                disc.save_weights('checkpoints/disc_%d.h5' % step)


if __name__ == '__main__':
    main()
