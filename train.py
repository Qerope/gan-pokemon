import argparse
import os

import numpy as np

import data
import models



def main():
    x = data.load_sprites()
    gen = models.build_generator()
    disc = models.build_discriminator()
    disc.compile('adam', 'binary_crossentropy')
    gan = models.build_gan(gen, disc)
    gan.compile('adam', 'binary_crossentropy')

    batch = 64
    step = 0
    for epoch in range(300):
        idx = np.random.permutation(len(x))
        for k in range(0, len(x) - batch + 1, batch):
            real = x[idx[k:k + batch]]
            z = np.random.normal(size=(batch, models.LATENT))
            fake = gen.predict(z)
            d_loss = disc.train_on_batch(real, np.ones((batch, 1)))
            d_loss += disc.train_on_batch(fake, np.zeros((batch, 1)))
            g_loss = gan.train_on_batch(z, np.ones((batch, 1)))
            step += 1
                print(step, d_loss, g_loss)


if __name__ == '__main__':
    main()
