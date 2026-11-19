#!/usr/bin/env bash
# Run from anywhere; -m needs the repo root as the working directory.
cd "$(dirname "$0")/.." || exit 1

# Realtime: you click the 4 sterile-field corners (top left, clockwise), then press q in the video window to stop.
# Everything is saved under SAVE_DIR.
SAVE_DIR=/home/mb230/projects/interactive_tutor/results/cldc/realtime_test

# python -m src.main_cldc \
#     --mode realtime \
#     --notification verbose \
#     --save-dir "$SAVE_DIR" \
#     --eval-phase passive \
#     --classifier-function vlm \
#     --human-profile-path /home/mb230/projects/interactive_tutor/data/cldc/human_01_profile.json

# Offline (later): replays the recorded videos in SAVE_DIR and reuses its sterile_field.pkl.
python -m src.main_cldc \
    --mode offline_mp4 \
    --notification verbose \
    --save-dir /home/mb230/projects/interactive_tutor/results/cldc/example_human1 \
    --eval-phase passive \
    --classifier-function vlm \
    --prompt-path /home/mb230/projects/interactive_tutor/src/domains/cldc/prompt.txt \
    --active-num-queries 1 \
    --active-num-epochs 100 \
    --human-profile-path /home/mb230/projects/interactive_tutor/data/cldc/human_01_profile.json \
    --error-report-path /home/mb230/projects/interactive_tutor/results/cldc/example_human1/error_dict.pkl
