import clip
import os
import sys
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="ViT-B/32")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--img_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)

args = parser.parse_args()

device = args.device
model_name = args.model
model, preprocess = clip.load(model_name, device=device)

# Assume the directory only contains file with suffix .jpg or .png ! Otherwise please modify line 40 as needed.
images = sorted(
   [os.path.join(args.img_dir, image) for image in list(os.listdir(args.img_dir))]
)

print(len(images))

for image in tqdm(images):
    image_obj = Image.open(image)
    images_obj = preprocess(image_obj).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images_obj).cpu().float().numpy()
        
    image_features = image_features[0,:]

    out_file = os.path.join(
        args.out_dir, os.path.basename(image)[:-4]
    )
    np.save(out_file, image_features)