""" The standalone script to prepare the modified version of annotation files.

This was used for generating results of re-implemented fully supervised methods.
"""
import json
import math
import os
import random

random.seed(0)


def generate_glance(glance_duration, start, end, precision=2):
    """ Generate the random glance from raw annotation.
    Args:
        glance_duration: number, use int for simplicity.
            The desired glance duration in seconds.
        start, end: float
            The timestamp (in seconds) read from raw annotation.
        precision: int
    """
    if end - start < glance_duration:
        glance_start = start
        glance_end = end
    else:
        glance_start = random.uniform(start, end - glance_duration)
        glance_end = glance_start + glance_duration
    return round(glance_start, precision), round(glance_end, precision)


def dump_activitynetcaptions(folder_path, filename, glance_duration):
    out_folder_path = os.path.join(folder_path, "glance{}".format(glance_duration))
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)

    with open(os.path.join(folder_path, filename), "r") as file:
        json_obj = json.load(file)
        for video_id in json_obj.keys():
            for i in range(len(json_obj[video_id]["timestamps"])):
                start, end = json_obj[video_id]["timestamps"][i]
                # fixing wrong annotation
                if start > end:
                    start, end = end, start
                glance_start, glance_end = generate_glance(glance_duration, start, end, precision=2)
                assert glance_start <= glance_end
                json_obj[video_id]["timestamps"][i] = [glance_start, glance_end]
        with open(os.path.join(out_folder_path, filename), "w") as out_file:
            json.dump(json_obj, out_file)


def dump_charadessta(folder_path, filename, glance_duration):
    out_folder_path = os.path.join(folder_path, "glance{}".format(glance_duration))
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
                glance_start, glance_end = generate_glance(glance_duration, start, end, precision=1)
                assert glance_start <= glance_end
                out_file.write("##".join([" ".join([video_id, str(glance_start), str(glance_end)]), annotation]))


def dump_tacos(folder_path, filename, glance_duration):
    out_folder_path = os.path.join(folder_path, "glance{}".format(glance_duration))
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)

    with open(os.path.join(folder_path, filename), "r") as file:
        json_obj = json.load(file)
        for video_id in json_obj.keys():
            for i in range(len(json_obj[video_id]["timestamps"])):
                start, end = json_obj[video_id]["timestamps"][i]
                fps = json_obj[video_id]["fps"]
                glance_duration_frame = glance_duration * fps

                glance_start, glance_end = generate_glance(glance_duration_frame, start, end, precision=1)
                glance_start, glance_end = math.floor(glance_start), math.floor(glance_end)
                assert glance_start <= glance_end
                json_obj[video_id]["timestamps"][i] = [glance_start, glance_end]
        with open(os.path.join(out_folder_path, filename), "w") as out_file:
            json.dump(json_obj, out_file)


if __name__ == '__main__':
    folder_path = "data/activitynetcaptions/annotations/"
    glance_durations = [3]
    for glance_duration in glance_durations:
        for filename in ["train.json", "val_1.json"]:
            dump_activitynetcaptions(folder_path, filename, glance_duration)

    folder_path = "data/charadessta/annotations"
    glance_durations = [3]
    for glance_duration in glance_durations:
        for filename in ["charades_sta_train.txt", "charades_sta_test.txt"]:
            dump_charadessta(folder_path, filename, glance_duration)

    folder_path = "data/tacos/annotations"
    glance_durations = [3]
    for glance_duration in glance_durations:
        for filename in ["train.json", "val.json"]:
            dump_tacos(folder_path, filename, glance_duration)
