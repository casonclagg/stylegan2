from shutil import copyfile
import torch
import clip
from PIL import Image
import sys
from pathlib import Path
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
categories = ["goofy", "soccer mom", "angry", "planet", "ironman"]

pathToSearch = sys.argv[1]
allProbs = []
for path in Path(pathToSearch).rglob('*.jpg'):
    try:
        filename = path #f"check{x+1:03}.jpg"
        image = preprocess(Image.open(filename)).unsqueeze(0).to(device)
        text = clip.tokenize(categories).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        thelist = probs[0].tolist()
        thelist.append(filename)
        allProbs.append(thelist)
    except:
        print(f"{filename} failed")
    # pedo = probs[0][0]
    # fat = probs[0][2]
    # normal = probs[0][1]
    # black = probs[0][3]
    # celeb = probs[0][4]

    # if(pedo > normal and pedo > fat and pedo > celeb and pedo > black): 
    #     print(f"{filename}: PEDO")
    # if(fat > normal and fat > pedo and fat > celeb and fat > black): 
    #     print(f"{filename}: FAT")
    # if(normal > pedo and normal > fat and normal > celeb and normal > black): 
    #     print(f"{filename}: NORMAL")
    # if(black > pedo and black > fat and black > celeb and black > normal): 
    #     print(f"{filename}: BLACK")
    # if(celeb > pedo and celeb > normal and celeb > black and celeb > fat): 
    #     print(f"{filename}: CELEB")
for x in range(len(categories)):
    sorted_by_second = sorted(allProbs, key=lambda tup: tup[x])
    print(f'Most {categories[x]}:')
    for y in range(20):
        el = sorted_by_second[len(sorted_by_second) - y - 1]
        src = el[len(el) - 1]
        print(src)
        copyfile(src, f'results/{categories[x]}_{y+1:03}.jpg')
    # print("Label probs:", f"pedo:{probs[0][0]}")  # prints: [[0.9927937  0.00421068 0.00299572]]
    # print("Label probs:", f"normal:{probs[0][1]}")
    # print("Label probs:", f"fat:{probs[0][2]}")