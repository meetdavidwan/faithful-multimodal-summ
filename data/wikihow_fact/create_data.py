import pickle
import json

data_dict = json.load(open("wikihow_text_by_step_id.json"))

from datasets import DatasetDict, Dataset

final_dataset = DatasetDict()

for split in ["random", "category", "similarity"]:
    print(split)
    for mode in ["test"]:
        dataset = {
            "id":[],
            "document":[],
            "image":[],
            "goal":[],
            "method":[],
            "step": [],
            "label": [],
        }
        data = pickle.load(open("{}_step_{}.p".format(mode, split),"rb"))

        for i, row in enumerate(data):
            step, images, correct_image_id = row
            correct_image = images[correct_image_id]

            dat = data_dict[correct_image]

            document = dat["description"]
            #summary = dat["headline"]

            steps = [data_dict[image]["headline"] for image in images]
            goals = [data_dict[image]["goal"] for image in images]
            methods = [data_dict[image]["method"] for image in images]

            for j, (step, goal, method) in enumerate(zip(steps, goals, methods)):
                dataset["id"].append("{}_{}".format(i, j))

                # correct instance
                dataset["image"].append(correct_image)
                dataset["document"].append(document)

                #dataset["image"].append(img)
                dataset["label"].append(1 if j == correct_image_id else 0)

                dataset["step"].append(step)
                dataset["goal"].append(goal)
                dataset["method"].append(method)

        dataset = Dataset.from_dict(dataset)
        final_dataset["{}_{}".format(mode, split)] = dataset

final_dataset.save_to_disk("wikihow_fact")