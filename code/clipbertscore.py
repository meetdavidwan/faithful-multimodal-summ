import clip
from bert_score import score
from tqdm import tqdm
import argparse
import os
import numpy as np
from PIL import Image
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--document", type=str, required=True)
parser.add_argument("--summary", type=str, required=True)
parser.add_argument("--image", type=str, required=True)
parser.add_argument("--img_dir", type=str)

parser.add_argument("--clip_model", type=str, default="RN50x64")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--preprocessed", action='store_true')

parser.add_argument("--alpha", type=float, default=0.25)

args = parser.parse_args()

documents = [line.strip() for line in open(args.document)]
summaries = [line.strip() for line in open(args.summary)]
images = [line.strip() for line in open(args.image)]

assert len(documents) == len(summaries) and len(summaries) == len(images)

# bertscore
bertscore , _, _ = score(summaries, documents, model_type="roberta-large-mnli", num_layers=10, verbose=False)
bertscore = bertscore.numpy()

# clipscore
model, preprocess = clip.load(args.clip_model, device=args.device)

clipscore = []
with torch.no_grad():
    for i in tqdm(range(0, len(args.image), args.batch_size)):
        summs = summaries[i:i+args.batch_size]

        # encode text
        text_features = clip.tokenize(summs, truncate=True).to(args.device)
        text_features = model.encode_text(text_features)

        # encode or load image
        image_features = []
        for img in images[i:i+args.batch_size]:
            if args.preprocessed:
                image_features.append(torch.from_numpy(np.load(os.path.join(args.img_dir, img))).unsqueeze(0).to(args.device))
            else:
                image_obj = Image.open(os.path.join(args.img_dir, img))
                images_obj = preprocess(image_obj).unsqueeze(0)
                image_features.append(model.encode_image(images_obj))
        image_features = torch.cat(image_features, dim=0)
    
        # normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if args.device == "cuda":
            image_features = image_features.half()

        logits_per_image = image_features @ text_features.t()

        for i in range(len(logits_per_image)):
            clipscore.append(logits_per_image[i,i].item())

clipscore = np.array(clipscore)


clipbertscore = args.alpha * clipscore + (1-args.alpha) * bertscore
print(clipbertscore)