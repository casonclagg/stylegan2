import subprocess
import os
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

def get_all_pickles(pickle_path):
    latent_files = [x for x in list(Path(pickle_path).iterdir()) if '.pkl' in x.name]
    latent_files.sort()
    return latent_files

def generate_image_from_z(z):
    images = Gs.run(z, None, **Gs_kwargs)
    return images
  
def generate_image_from_pickle(latent_vector):
    images = Gs.components.synthesis.run(latent_vector, **Gs_kwargs)
    return images

def average(pickleA, pickleB, name, outputDir):
    save_dir = Path(f"results/{outputDir}/{name}")
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
    PIL.Image.fromarray(images[0], 'RGB').save((f'results/{outputDir}/{name}/average.png'))
    with open(f'results/{outputDir}/{name}/average.pkl', 'wb') as out_file:
        pickle.dump(latent_a, out_file)

def bang(pickleA, pickleB, name, index, outputDir):
    with open(pickleA, mode='rb') as latent_pickle:
        latent_a = pickle.load(latent_pickle)
    with open(pickleB, mode='rb') as latent_pickle:
        latent_b = pickle.load(latent_pickle)
    for z_index in range(512):
        # if random.random() > 0.5:
        small = min([latent_b[0][0][z_index],latent_a[0][0][z_index]])
        big = max([latent_b[0][0][z_index],latent_a[0][0][z_index]])
        latent_a[0][0][z_index] =  random.uniform(small, big)
        for x in range(18):
            latent_a[0][x][z_index] = latent_a[0][0][z_index]
    images = generate_image_from_pickle(latent_a)
    PIL.Image.fromarray(images[0], 'RGB').save((f'results/{outputDir}/{name}/kid-{index:05}.png'))
    with open(f'results/{outputDir}/{name}/kid-{index:05}.pkl', 'wb') as out_file:
        pickle.dump(latent_a, out_file)

def parents(outputDir, dad, mom, name):
    save_dir = Path(f"./results/{outputDir}")
    if not save_dir.exists():
        save_dir.mkdir()

    outtie = Path(f"./results/{outputDir}/{name}")
    if not outtie.exists():
        outtie.mkdir()

    with open(dad, mode='rb') as latent_pickle:
        latent_a = pickle.load(latent_pickle)
        print(type(latent_a))
    with open(mom, mode='rb') as latent_pickle:
        latent_b = pickle.load(latent_pickle)

    images = generate_image_from_pickle(latent_a)
    PIL.Image.fromarray(images[0], 'RGB').save((f'results/{outputDir}/{name}/dad.png'))
    
    images = generate_image_from_pickle(latent_b)
    PIL.Image.fromarray(images[0], 'RGB').save((f'results/{outputDir}/{name}/mom.png'))

    average(dad, mom, name, outputDir)

# python breed-alot.py pickles babes 20
def main():
    pickleDir = sys.argv[1]
    outputDir = sys.argv[2]
    babyCount = int(sys.argv[3])

    all_pickles = get_all_pickles(pickleDir)
    
    count = 0
    for dad in all_pickles:
        for mom in all_pickles:
            count += 1
            # if(count > 5) quit()

            dad_name = os.path.splitext(dad.name)[0]
            mom_name = os.path.splitext(mom.name)[0]
            parents_name = f"{dad_name}x{mom_name}"
            if dad_name == mom_name:
                continue

            parents(outputDir, dad, mom, parents_name)

            for x in range(babyCount):
                bang(dad, mom, parents_name, x, outputDir)
            
            # convert dad.png average.png -size 1024x50 xc:White +swap -background White -gravity South +append temp.png
            # convert temp.png mom.png -size 1024x50 xc:White +swap -background White -gravity South +append parents.png
            # montage -mode concatenate -tile 5x "kid*.png" kids.jpg
            # convert parents.png kids.png -gravity center -append final.jpg
            subprocess.run(["convert",f"results/{outputDir}/{parents_name}/dad.png",f"results/{outputDir}/{parents_name}/average.png","-size","1024x50","xc:White","+swap","-background","White","-gravity","South","+append",f"results/{outputDir}/{parents_name}/temp.png"])
            subprocess.run(["convert",f"results/{outputDir}/{parents_name}/temp.png",f"results/{outputDir}/{parents_name}/mom.png","-size","1024x50","xc:White","+swap","-background","White","-gravity","South","+append",f"results/{outputDir}/{parents_name}/parents.png"])
            subprocess.run(["montage", "-mode", "concatenate", "-tile", "5x", f"results/{outputDir}/{parents_name}/kid*.png", f"results/{outputDir}/{parents_name}/kids.png"])
            subprocess.run(["convert",f"results/{outputDir}/{parents_name}/parents.png",f"results/{outputDir}/{parents_name}/kids.png","-gravity","center","-append",f"results/{outputDir}/{parents_name}.png"])
    
if __name__ == "__main__":
    main()