from pathlib import Path
import  random
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

def average(pickleA, pickleB, name):
    save_dir = Path(f"results/babies/{name}")
    if not save_dir.exists():
        save_dir.mkdir()
  
    with open(pickleA, mode='rb') as latent_pickle:
        latent_a = pickle.load(latent_pickle)
    with open(pickleB, mode='rb') as latent_pickle:
        latent_b = pickle.load(latent_pickle)
        
    for z_index in range(512):
        latent_a[0][0][z_index] = (latent_a[0][0][z_index] + latent_b[0][0][z_index]) / 2
        for x in range(18):
            latent_a[0][x][z_index] = latent_a[0][0][z_index]
    images = generate_image_from_pickle(latent_a)
    PIL.Image.fromarray(images[0], 'RGB').save((f'results/babies/{name}/average.png'))
    with open(f'results/babies/{name}/average.pkl', 'wb') as out_file:
        pickle.dump(latent_a, out_file)

def bang(pickleA, pickleB, name, index):
    with open(pickleA, mode='rb') as latent_pickle:
        latent_a = pickle.load(latent_pickle)
    with open(pickleB, mode='rb') as latent_pickle:
        latent_b = pickle.load(latent_pickle)
    for z_index in range(512):
        if random.random() > 0.5:
            latent_a[0][0][z_index] =  latent_b[0][0][z_index]
        for x in range(18):
            latent_a[0][x][z_index] = latent_a[0][0][z_index]
    images = generate_image_from_pickle(latent_a)
    PIL.Image.fromarray(images[0], 'RGB').save((f'results/babies/{name}/{index:05}.png'))
    with open(f'results/babies/{name}/{index:05}.pkl', 'wb') as out_file:
        pickle.dump(latent_a, out_file)
        
# python breed.py c3c4 "results/pickles/cason-2.pkl" "results/pickles/cason-3.pkl" 25
def main():
    name = sys.argv[1]
    pickleA = sys.argv[2]
    pickleB = sys.argv[3]
    babyCount = int(sys.argv[4])
    
    save_dir = Path(f"./results/babies/{name}")
    if not save_dir.exists():
        save_dir.mkdir()

    with open(pickleA, mode='rb') as latent_pickle:
        latent_a = pickle.load(latent_pickle)
    with open(pickleB, mode='rb') as latent_pickle:
        latent_b = pickle.load(latent_pickle)

    images = generate_image_from_pickle(latent_a)
    PIL.Image.fromarray(images[0], 'RGB').save((f'results/babies/{name}/a-dad.png'))
    
    images = generate_image_from_pickle(latent_b)
    PIL.Image.fromarray(images[0], 'RGB').save((f'results/babies/{name}/a-mom.png'))

    average(pickleA, pickleB, name)

    for x in range(babyCount):
        bang(pickleA, pickleB, name, x)
    
if __name__ == "__main__":
    main()