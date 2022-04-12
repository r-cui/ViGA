""" The standalone script to prepare the modified version of annotation files under the glance annotation setting.

Note that even if a random seed is set, the result is likely to be different on different machines. Hence,
preferably not to run this script but use the data already generated. We keep this script for reference only.
"""
import json
import math
import os
import random

random.seed(0)


def generate_glance(start, end, precision=2):
    """ Generate the random glance from raw annotation.
    Args:
        glance_duration: number, use int for simplicity.
            The desired glance duration in seconds.
        start, end: float
            The timestamp (in seconds) read from raw annotation.
        precision: int
    """
    glance = random.uniform(start, end)
    return round(glance, precision)


def dump_activitynetcaptions(folder_path, filename):
    out_folder_path = os.path.join(folder_path, "glance")
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)

    with open(os.path.join(folder_path, filename), "r") as file:
        json_obj = json.load(file)
        for video_id in json_obj.keys():
            temp = []
            for i in range(len(json_obj[video_id]["timestamps"])):
                start, end = json_obj[video_id]["timestamps"][i]
                # fixing wrong annotation
                if start > end:
                    start, end = end, start
                temp.append(generate_glance(start, end, precision=2))
            json_obj[video_id]["glance"] = temp
        with open(os.path.join(out_folder_path, filename), "w") as out_file:
            json.dump(json_obj, out_file)


def dump_charadessta(folder_path, filename):
    out_folder_path = os.path.join(folder_path, "glance")
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)

    with open(os.path.join(out_folder_path, filename), "w") as out_file:
        with open(os.path.join(folder_path, filename), "r") as file:
            lines = file.readlines()
            for line in lines:
                video_id, start, end = line.split("##")[0].split()
                start, end = float(start), float(end)
                # fixing wrong annotation
                if start > end:
                    start, end = end, start
                annotation = line.split("##")[1]
                glance = generate_glance(start, end, precision=1)
                out_file.write("##".join([" ".join([video_id, str(start), str(end)]), str(glance), annotation]))


def dump_tacos(folder_path, filename):
    out_folder_path = os.path.join(folder_path, "glance")
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)

    with open(os.path.join(folder_path, filename), "r") as file:
        json_obj = json.load(file)
        for video_id in json_obj.keys():
            temp = []
            for i in range(len(json_obj[video_id]["timestamps"])):
                start, end = json_obj[video_id]["timestamps"][i]
                glance = generate_glance(start, end, precision=0)
                temp.append(int(glance))
            json_obj[video_id]["glance"] = temp
        with open(os.path.join(out_folder_path, filename), "w") as out_file:
            json.dump(json_obj, out_file)


if __name__ == '__main__':
    folder_path = "data/activitynetcaptions/annotations/"
    for filename in ["train.json", "val_1.json", "val_2.json"]:
        dump_activitynetcaptions(folder_path, filename)

    folder_path = "data/charadessta/annotations"
    for filename in ["charades_sta_train.txt", "charades_sta_test.txt"]:
        dump_charadessta(folder_path, filename)

    folder_path = "data/tacos/annotations"
    for filename in ["train.json", "val.json", "test.json"]:
        dump_tacos(folder_path, filename)
