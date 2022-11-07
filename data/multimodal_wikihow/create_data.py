import json
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict

data_dict = json.load(open("wikihow_text_by_step_id.json"))

possible_images = [image.strip() for image in open("images.txt")] + [image.strip() for image in open("test_images.txt")]

valid_ids = set()

id2steps = defaultdict(list)

for image in possible_images:
    image_step = image.split("_")[-1]
    image = "_".join(image.split("_")[:2])
    valid_ids.add(image)

    id2steps[image].append(image_step)

for k, v in id2steps.items():
    id2steps[k] = sorted(v, key=lambda x:int(x))


valid_ids = list(valid_ids)

X_train, X_test = train_test_split(valid_ids, test_size=12000, random_state=42)
X_valid, X_test = train_test_split(X_test, test_size=0.5, random_state=42)

from datasets import Dataset, DatasetDict

ids = {"train": X_train, "valid": X_valid, "test": X_test}

dataset_dict = DatasetDict()

for k,v in ids.items():
    ddict = {"image":[],"document":[], "summary":[] }
    ids = []
    for id in v:
        for step_id in id2steps[id]:
            full_id = "{}_{}".format(id, step_id)
            data = data_dict[full_id]
            ids.append(full_id)

            ddict["image"].append(full_id)
            ddict["document"].append(data["description"])
            ddict["summary"].append(data["headline"])

    dataset_dict[k] = Dataset.from_dict(ddict)

    # if k == "test":
    #     json.dump(ids, open("test_ids.json","w"))

dataset_dict.save_to_disk("wikihow_summarization")