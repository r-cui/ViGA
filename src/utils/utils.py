import csv
import json
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

from datetime import datetime
from pathlib import Path
from tqdm import tqdm


def get_now():
    """ Get a string representation of current time, for distinguishing different runs.
    """
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def safe_division(numerator, denominator, precision=4):
    """ Use in getting "start_frac" and "end_frac" when creating dataloader objects.

    Args:
        numerator, denominator: can be in frame (int) or timestamp (float), as long as consistent
    Returns:
        float
    """
    return round(max([min([numerator / denominator, 1.0]), 0.0]), precision)


# torch utils
def n_params(model):
    """ Calculate total number of parameters in a model.
    Args:
        model: nn.Module
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sliding_window(x, window_size, stride, dim=1):
    """
    Args:
        x: (B, L, dim)
    Returns:
        splits: [(B, window_size, dim)] with last element might be shorter.
        idx: (L, 2)
    """
    start = torch.arange(0, x.shape[dim] - window_size + 1, stride, device=x.device)
    end = torch.arange(window_size, x.shape[dim] + 1, stride, device=x.device)
    idx = torch.stack((start, end), dim=1)

    splits = [
        torch.index_select(
            x, dim, torch.arange(idx[i, 0], idx[i, 1], device=x.device)
        ) for i in range(idx.shape[0])
    ]
    return splits, idx


# IO utils
def load_config(yaml_dir):
    with open(yaml_dir) as f:
        res = yaml.safe_load(f)

    # build directory if does not exist
    Path(res["exp_dir"]).mkdir(parents=True, exist_ok=True)
    return res


# dataloader utils
def load_annotations_activitynetcaptions(split):
    split2filename = {
        "train": "train.json",
        "valid": "val_1.json",
        "test": "val_2.json"
    }

    annotation_folder_path = os.path.join("data", "activitynetcaptions", "annotations", "glance")

    annotations = {}

    with open(os.path.join(annotation_folder_path, split2filename[split]), "r") as annotation_file:
        json_obj = json.load(annotation_file)
        idx = 0
        for video_id in tqdm(json_obj.keys(), desc="Loading ActivityNet Captions {} annotations".format(split)):
            video_anno = json_obj[video_id]
            for i in range(len(video_anno["timestamps"])):
                annotation = video_anno["sentences"][i]
                annotations[idx] = {
                    "idx": idx,
                    "video_id": video_id,
                    "duration": video_anno["duration"],

                    "annotation": annotation,

                    "start_frac": safe_division(video_anno["timestamps"][i][0], video_anno["duration"]),
                    "end_frac": safe_division(video_anno["timestamps"][i][1], video_anno["duration"]),
                    "glance_frac": safe_division(video_anno["glance"][i], video_anno["duration"])
                }
                idx += 1
    return annotations


def load_annotations_charadessta(split):
    def _load_durations(split):
        """ Read Charades_v1_train.csv / Charades_v1_test.csv and load video durations.
        Returns:
            {str: float}
        """
        split2filename = {
            "train": "Charades_v1_train.csv",
            "valid": "Charades_v1_test.csv",
            "test": "Charades_v1_test.csv"
        }
        annotation_folder_path = os.path.join("data", "charadessta", "annotations")

        with open(os.path.join(annotation_folder_path, split2filename[split]), "r") as annotation_info_file:
            csv_reader = csv.reader(annotation_info_file, delimiter=',')
            first_line_flag = True
            durations = dict()
            for row in csv_reader:
                if not first_line_flag:
                    durations[row[0]] = float(row[-1])
                first_line_flag = False
        return durations

    durations = _load_durations(split)

    split2filename = {
        "train": "charades_sta_train.txt",
        "valid": "charades_sta_test.txt",
        "test": "charades_sta_test.txt"
    }
    annotation_folder_path = os.path.join("data", "charadessta", "annotations", "glance")

    annotations = {}
    with open(os.path.join(annotation_folder_path, split2filename[split]), "r") as annotation_file:
        lines = annotation_file.readlines()
        for i in tqdm(range(len(lines)), desc="Loading Charades-STA {} annotations".format(split)):
            line = lines[i]
            video_id, start, end = line.split("##")[0].split()
            start, end = float(start), float(end)
            glance = float(line.split("##")[1])
            annotation = line.split("##")[2].rstrip()

            annotations[i] = {
                "idx": i,
                "video_id": video_id,
                "duration": durations[video_id],

                "annotation": annotation,

                "start_frac": safe_division(start, durations[video_id]),
                "end_frac": safe_division(end, durations[video_id]),
                "glance_frac": safe_division(glance, durations[video_id])
            }
    return annotations


def load_annotations_tacos(split):
    split2filename = {
        "train": "train.json",
        "valid": "val.json",
        "test": "test.json"
    }
    annotation_folder_path = os.path.join("data", "tacos", "annotations", "glance")

    annotations = {}

    with open(os.path.join(annotation_folder_path, split2filename[split]), "r") as annotation_file:
        json_obj = json.load(annotation_file)
        idx = 0
        for video_id in tqdm(json_obj.keys(), desc="Loading TACoS {} annotations".format(split)):
            video_anno = json_obj[video_id]
            for i in range(len(video_anno["timestamps"])):
                duration = round(video_anno["num_frames"] / video_anno["fps"], 2)
                annotation = video_anno["sentences"][i]

                annotations[idx] = {
                    "idx": idx,
                    "video_id": video_id[:-4],  # remove .avi
                    "duration": duration,

                    "annotation": annotation,

                    "start_frac": safe_division(video_anno["timestamps"][i][0], video_anno["num_frames"]),
                    "end_frac": safe_division(video_anno["timestamps"][i][1], video_anno["num_frames"]),
                    "glance_frac": safe_division(video_anno["glance"][i], video_anno["num_frames"])
                }
                idx += 1
    return annotations
