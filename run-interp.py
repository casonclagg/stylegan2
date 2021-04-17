import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import pretrained_networks
from tqdm import tqdm

network_pkl = "gdrive:networks/stylegan2-ffhq-config-f.pkl"
print('Loading networks from "%s"...' % network_pkl)
_G, _D, Gs = pretrained_networks.load_networks(network_pkl)
noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

Gs_kwargs = dnnlib.EasyDict()
Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
Gs_kwargs.randomize_noise = False

def generate_image_from_z(z):
    images = Gs.run(z, None, **Gs_kwargs)
    return images

def linear_interpolate(code1, code2, alpha):
    return code1 * alpha + code2 * (1 - alpha)

def make_latent_interp_animation(code1, code2, num_interps):
    step_size = 1.0/num_interps
    amounts = np.arange(0, 1, step_size)
    count = 0
    for alpha in tqdm(amounts):
        interpolated_latent_code = linear_interpolate(code1, code2, alpha)
        images = generate_image_from_z(interpolated_latent_code)
        PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path(f'{count}.png'))
        count += 1

def generate(steps):
    rnd = np.random.RandomState(1)
    z1 = rnd.randn(1, *Gs.input_shape[1:])
    z2 = rnd.randn(1, *Gs.input_shape[1:])

    make_latent_interp_animation(z1,z2,steps)

def main():
    generate(10)

if __name__ == "__main__":
    main()