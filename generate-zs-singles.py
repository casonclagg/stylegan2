import math
import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import pretrained_networks
from tqdm import tqdm
import pickle

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
  
def make_latent_interp_animation(z_index, z, change, index):
    # step_size = math.pi * 2 / num_interps
    # amounts = np.arange(0, math.pi * 2, step_size)
    # count =  num_interps * z_index
    
    og_value = z[0][0][z_index]

    val = og_value + change
    for x in range(18):
        z[0][x][z_index] = val
    images = generate_image_from_pickle(z)
    PIL.Image.fromarray(images[0], 'RGB').save((f'results/zs-singles/{z_index:06}.{index:03}.png'))
        
# python generate-zs-singles.py "results/pickles/cason-4.pkl"
def main():
    pickleFile = sys.argv[1]
    # numberOfFrames = int(sys.argv[2])
    
    start_z = []
    print(f'loading {pickleFile}')

    for z_index in range(512):
        values_to_do = [-50,-30,-20, -10, -5, 0, 5, 10, 20, 30, 50]
        for index in range(len(values_to_do)):
            change = values_to_do[index]
            print(f"doing {z_index} now, {index} and {change}")
            with open(pickleFile, mode='rb') as latent_pickle:
                start_z = pickle.load(latent_pickle)
            og_z = []
            for x in range(512):
                og_z.append(start_z[0][0][x])
            for x in range(18):
                for xx in range(512):
                    start_z[0][x][xx] = start_z[0][0][xx]
            make_latent_interp_animation(z_index, start_z, change, index)
    
if __name__ == "__main__":
    main()
    
def get_final_latents():
    all_results = list(Path('results/').iterdir())
    all_results.sort()
    last_result = all_results[-1]
    latent_files = [x for x in last_result.iterdir() if 'final_latent_code' in x.name]
    latent_files.sort()
    all_final_latents = []
    for file in latent_files:
        with open(file, mode='rb') as latent_pickle:
            all_final_latents.append(pickle.load(latent_pickle))
    return all_final_latents