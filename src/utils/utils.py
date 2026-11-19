import argparse
import random

import numpy as np
import torch
from PIL import Image
from skimage.util import montage


def parse_args():
    # 'verbose' draws the skeletal tracks and rule violations on the display; 'silent' shows the live feed without them
    parser = argparse.ArgumentParser()
    parser.add_argument("--notification", choices=["verbose", "silent"], default="verbose")
    parser.add_argument("--mode",choices=["realtime", "offline_mp4"], default="realtime")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--human-profile-path", type=str, required=True)
    parser.add_argument("--eval-phase", choices = ["passive", "active", "tutor"], type=str, required=True)
    parser.add_argument("--model-path", type=str, default="/home/mb230/projects/adaptive_skill/assets/mediapipe_models/pose_landmarker.task")
    parser.add_argument("--hand-model-path", type=str, default="/home/mb230/projects/adaptive_skill/assets/mediapipe_models/hand_landmarker.task")
    parser.add_argument("--error-report-path", type=str, default=None)
    parser.add_argument("--active-query-mode", type=str, default="random")
    parser.add_argument("--active-num-queries", type=int, default=5)
    parser.add_argument("--active-num-epochs", type=int, default=10)
    parser.add_argument("--classifier-function", choices=["linear", "vlm"], default="linear")
    # text file holding the prompt for the VLM classifier (only used with --classifier-function vlm)
    parser.add_argument("--prompt-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    # also save per-frame landmarks to pose.npz
    parser.add_argument("--save-pose", action="store_true", default=False)

    return parser.parse_args()


def set_global_seed(seed: int = 0):
    """Seed python, numpy and torch (CPU + CUDA) for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def oh_encode_categories(cat_idx, tot_cat):
    """Return a one-hot numpy vector of length tot_cat with a 1 at cat_idx."""
    if not 0 <= cat_idx < tot_cat:
        raise ValueError(f"cat_idx {cat_idx} out of range for {tot_cat} categories")
    encoding = np.zeros(tot_cat, dtype=np.float32)
    encoding[cat_idx] = 1.0
    return encoding


def make_montage(image_list, montage_save_path, montage_dims):
    """Tile RGB PIL images into a (rows, cols) grid and save it to montage_save_path."""
    grid = montage(np.stack([np.asarray(img) for img in image_list]), grid_shape=montage_dims, channel_axis=-1)
    Image.fromarray(grid).save(montage_save_path)
    return grid
