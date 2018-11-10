# gan-pokemon

DCGAN on pokemon sprites.

Put 64x64 sprite PNGs in `data/` (I used the gen 1-5 front sprites, resized),
then:

    python train.py --epochs 300
    python train.py --resume checkpoints/gen_5000.h5

A sample grid is written to `samples/` every 200 steps. It produces blobs with
sensible colour palettes after about 3000 steps and recognisable body shapes
after ~10k; it collapses to a handful of shapes if the discriminator gets
ahead, which is why there is label noise.
