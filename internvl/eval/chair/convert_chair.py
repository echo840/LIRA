import json
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
def setup_seeds(config):
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

if __name__=="__main__":
    json_save = []
    img_files = os.listdir("data/coco/val2014")
    random.shuffle(img_files)

    with open('data/chair/annotations/instances_val2014.json', 'r') as f:
        lines = f.readlines()
    coco_anns = json.loads(lines[0])

    img_dict = {}

    categories = coco_anns["categories"]
    category_names = [c["name"] for c in categories]
    category_dict = {int(c["id"]): c["name"] for c in categories}

    for img_info in coco_anns["images"]:
        img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}

    for ann_info in coco_anns["annotations"]:
        img_dict[ann_info["image_id"]]["anns"].append(
            category_dict[ann_info["category_id"]]
        )

    for img_id in tqdm(range(len(img_files))):
        if img_id == 500:
            break
        img_file = img_files[img_id]
        img_id = int(img_file.split(".jpg")[0][-6:])
        img_info = img_dict[img_id]
        assert img_info["name"] == img_file
        img_anns = set(img_info["anns"])
        img_save = {}
        img_save["image_id"] = img_id

        image_path = "data/coco/val2014/" + img_file
        qu = "Please describe this image in detail."
        json_save.append({"image":image_path,"question":qu,"question_id": img_id, "answer": ""})
    with open('data/chair/chair_test.jsonl', 'w') as f:
        for entry in json_save:
            f.write(json.dumps(entry) + '\n')