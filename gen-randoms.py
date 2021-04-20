import pickle
from pathlib import Path
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
print(f'Loading networks from {network_pkl}...')
_G, _D, Gs = pretrained_networks.load_networks(network_pkl)
noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
Gs_kwargs = dnnlib.EasyDict()
Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
Gs_kwargs.randomize_noise = False

def generate_image_from_z(z):
    images = Gs.run(z, None, **Gs_kwargs)
    return images

def generate_image_from_pickle(latent_vector):
    images = Gs.components.synthesis.run(latent_vector, **Gs_kwargs)
    return images

def linear_interpolate(code1, code2, alpha):
    return code1 * alpha + code2 * (1 - alpha)
  
def make_latent_interp_animation(code1, code2, num_interps, index):
    step_size = 1.0/num_interps
    amounts = np.arange(0, 1, step_size)
    count =  num_interps * index
    for alpha in tqdm(amounts):
        interpolated_latent_code = linear_interpolate(code1, code2, alpha)
        images = generate_image_from_pickle(interpolated_latent_code)
        PIL.Image.fromarray(images[0], 'RGB').save(f'results/{count:05}.png')
        count += 1
        
def generate(steps):
    rnd = np.random.RandomState(1)
    z1 = rnd.randn(1, *Gs.input_shape[1:])
    z2 = rnd.randn(1, *Gs.input_shape[1:])
    make_latent_interp_animation(z1,z2,steps)

def get_final_latents():
    # all_results = list(Path('results/pickles').iterdir())
    # all_results.sort()
    # last_result = all_results[-1]
    # latent_files = [x for x in Path('results/pickles').iterdir() if 'final_latent_code' in x.name]
    latent_files = [x for x in Path('results/pk2').iterdir()]
    latent_files.sort()
    all_final_latents = []
    for file in latent_files:
        print(f'loading {file}')
        with open(file, mode='rb') as latent_pickle:
            all_final_latents.append(pickle.load(latent_pickle))
    return all_final_latents

def main():
    rnd = np.random.RandomState(1)
    # with open(f"results/randoms/00000.pkl", mode='rb') as latent_pickle:
    #     z = pickle.load(latent_pickle)
    #     images = generate_image_from_z(z)
    #     PIL.Image.fromarray(images[0], 'RGB').save(f'results/randoms/TESTING.png')
    for x in range(500):
        z = rnd.randn(1, *Gs.input_shape[1:])
        images = generate_image_from_z(z)
        PIL.Image.fromarray(images[0], 'RGB').save(f'results/randoms/{x:05}.png')
        with open(f'results/randoms/{x:05}.pkl', 'wb') as out_file:
            pickle.dump(z, out_file)
    
if __name__ == "__main__":
    main()
    
