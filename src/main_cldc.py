import hydra
import cv2
import pickle
import numpy as np
from PIL import Image
from omegaconf import DictConfig
import os
import sys
import json
import time
from src.domains.cldc.sterile_field import (
    SterileField,
    draw_rules,
    draw_sterile_field,
    rule_to_idx,
)
from src.reader import Reader
from src.utils.mediapipe_utils import MediaPipe, colorize_depth, DEPTH_VIS_MAX
from src.utils.utils import parse_args, set_global_seed
from src.domains.cldc.error_categorization import (
    PassiveClassifier,
    PassiveClassifierVLM,
    passive_categorization,
    active_categorization,
)
import torch.nn.functional as F
from src.utils.utils import oh_encode_categories, make_montage
import random
import torch

# below are some config parameters
here = os.path.dirname(os.path.abspath(__file__))


class CLDCEvaluator:
    NUM_CONTEXT_BEFORE = 10  # context for the VLM = 10 frames before + violating frame + 10 after = 21
    NUM_CONTEXT_AFTER = 10

    def __init__(self, args):
        self.args = args
        set_global_seed(args.seed)

        self.fps = 30 if args.mode == "realtime" else 15
        with open(args.human_profile_path) as f:
            self.human_profile = json.load(f)
        # one folder per rule; violating frames are saved as <rule>/<frame_idx>.png
        self.out_path = os.path.join(args.save_dir, "pose.npz")
        for rule in rule_to_idx:
            os.makedirs(os.path.join(args.save_dir, rule), exist_ok=True)

        self.video_path = os.path.join(args.save_dir, f"{args.mode}.mp4")

        self.idx_to_rule = {1: "hbt", 2: "ra", 3: "tb", 4: "oib"}
        os.makedirs(args.save_dir, exist_ok=True)
        self.error_dict = {}

        # the two modes differ only in where frames come from (and whether a window is shown)
        if args.mode == "realtime":
            self.reader = Reader("realtime")
            self.sterile_field = SterileField()
            self.sterile_field.detect_manual("realtime", "rgb", reader=self.reader)
            # save sterile field
            sterile_field_path = os.path.join(args.save_dir, "sterile_field.pkl")
            with open(sterile_field_path, "wb") as file:
                pickle.dump(self.sterile_field, file)
            print(
                "sterile field created successfully!"
                + (
                    " Press q in the video window to quit."
                    if args.mode == "realtime"
                    else ""
                )
            )
            print(f"saved sterile field at {sterile_field_path}")

        elif args.mode == "offline_mp4":
            self.reader = Reader(args.mode, path=args.save_dir)
            sterile_field_load_path = os.path.join(args.save_dir, "sterile_field.pkl")
            self.sterile_field = pickle.load(open(sterile_field_load_path, "rb"))
            print(f"loaded sterile field from {sterile_field_load_path}")
            print(
                f"Total number of frames in the reader is {self.reader.num_frames()}"
            )

        else:
            raise ValueError(f"Mode {args.mode} not implemented/defined yet!")

        self.frames = self.reader.frames()

        self.mediapipe_obj = MediaPipe(
            args.model_path,
            args.hand_model_path,
            args.min_detection_confidence,
            args.min_tracking_confidence,
        )

        self.viol_frames = {rule: 0 for rule in rule_to_idx}
        self.depth_frames = []
        self.ts, self.lms_cam = [], []  # only filled if args.save_pose

        if args.mode == "offline_mp4":
            print(f"Total number of frames in recorded video is {self.reader.num_frames()}")
            self.dur_ms = int(self.reader.num_frames() * 1000 / self.fps)
            print(f"Total video duration is {self.dur_ms} ms")

        self.feature_dim = (
            len(rule_to_idx) + 3
        )  # for now we are assuming the scenario of no interventions and just a single phase in the task

        # the whole raw video is decoded once (lazily) and sliced for every violation's context frames (VLM only)
        self.video_frames = None

        if args.classifier_function == "linear":
            self.passive_clf = PassiveClassifier(
                num_classes=3, feature_dim=self.feature_dim
            )
        elif args.classifier_function == "vlm":
            with open(args.prompt_path, "r") as f:
                self.base_prompt = f.read().strip()
            self.passive_clf = PassiveClassifierVLM()
        else:
            raise ValueError(f"{args.classifier_function} is not defined!")

    def load_video_frames(self):
        """Decode the untouched recording (realtime_raw.mp4) into a list of BGR frames, once."""
        cap = cv2.VideoCapture(self.reader.mp4_rgb_path)
        assert cap.isOpened(), f"could not open {self.reader.mp4_rgb_path}"
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        return frames

    def get_context_frames(self, frame_index, before_num, after_num):
        """Return the BGR frames [frame_index - before_num, frame_index + after_num], clamped to the video bounds.

        Needs the complete recording, so it only works in offline_mp4 mode (in realtime the "after" frames don't exist yet).
        """
        if self.args.mode != "offline_mp4":
            raise NotImplementedError(
                "context frames need the full recording, so they are only available in offline_mp4 mode"
            )
        
        if self.video_frames is None:
            self.video_frames = self.load_video_frames()

        start = max(0, frame_index - before_num)
        end = min(len(self.video_frames), frame_index + after_num + 1)
        return self.video_frames[start:end]



    def eval_cldc(self):

        num_detected = 0
        writer = None
        raw_writer = None
        depth_writer = None

        if self.args.eval_phase == "passive":
            print("doing passive evaluation")
            if self.args.mode == "realtime":
                cv2.namedWindow(
                    "realtime", cv2.WINDOW_NORMAL
                )  # WINDOW_NORMAL lets you drag the window corners to resize
                # saved once, before the loop; Reader's height_color/width_color are swapped, so the frame is (width=height_color, height=width_color)
                with open(
                    os.path.join(self.args.save_dir, "realtime_depth_info.json"), "w"
                ) as f:
                    json.dump(
                        dict(
                            fx=self.reader.new_camera_matrix[0, 0],
                            fy=self.reader.new_camera_matrix[1, 1],
                            ppx=self.reader.new_camera_matrix[0, 2],
                            ppy=self.reader.new_camera_matrix[1, 2],
                            width=self.reader.height_color,
                            height=self.reader.width_color,
                            distortion="none (color and depth both undistorted with new_camera_matrix)",
                            depth_scale_m_per_count=self.reader.depth_scale,
                            colormap="JET",
                            colormap_range_m=[0.0, DEPTH_VIS_MAX],
                            no_measurement="black (0,0,0)",
                        ),
                        f,
                        indent=2,
                    )
                # frame size from the reader (swapped, see above); the raw and depth videos share size and fps, so frame i of both is the same instant and pixel
                w, h = self.reader.height_color, self.reader.width_color
                raw_writer = cv2.VideoWriter(
                    os.path.join(self.args.save_dir, "realtime_raw.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (w, h),
                )
                depth_writer = cv2.VideoWriter(
                    os.path.join(self.args.save_dir, "realtime_depth.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (w, h),
                )
            try:
                for i, (_, frame_bgr, depth_raw) in enumerate(self.frames):
                    print(f"Processing frame {i}")
                    if (
                        self.args.mode != "realtime"
                    ):  # realtime already got w, h from the reader above
                        h, w = frame_bgr.shape[:2]
                    t_ms = int(i * 1000 / self.fps)

                    self.mediapipe_obj.estimate_pose(frame_bgr, depth_raw, self.reader, t_ms)
                    self.mediapipe_obj.estimate_hands(frame_bgr, depth_raw, self.reader, t_ms)

                    if self.args.save_pose:
                        self.ts.append(t_ms)
                        self.lms_cam.append(
                            np.array(
                                [
                                    np.full(3, np.nan) if p is None else p.unpack()
                                    for p in self.mediapipe_obj.pose_keypoints_camera_coordinates
                                ]
                            )
                        )

                    if (
                        self.args.mode == "realtime"
                    ):  # save the untouched frame before any overlay is drawn
                        raw_writer.write(frame_bgr)
                        depth_writer.write(
                            colorize_depth(depth_raw * self.reader.depth_scale)
                        )
                        self.depth_frames.append(depth_raw)

                    violated = []

                    # for now the intervention and task phase both are just 0, i.e they are not used, but this is something that can be inferred from the video later on for such active sample selection
                    intervention = 0
                    task_phase = 0

                    contaminations = self.sterile_field.is_contaminated(self.mediapipe_obj)

                    if not contaminations[0]:  # index 0 = no human found
                        num_detected += 1
                        for rule, idx in rule_to_idx.items():
                            if contaminations[idx]:
                                self.viol_frames[rule] += 1
                                violated.append(rule)

                        self.error_dict[i] = {}

                        for rule, idx in rule_to_idx.items():
                            if contaminations[idx]:
                                feat_vec = np.concatenate(
                                    [
                                        oh_encode_categories(idx - 1, len(rule_to_idx)),
                                        [t_ms / self.dur_ms if self.args.mode == "offline_mp4" else 0, 0, 0],
                                    ]
                                ).astype(np.float32)

                                if self.args.classifier_function == "linear":
                                    context_frames = None
                                elif self.args.classifier_function == "vlm":
                                    # 21 frames: 10 before, the violating frame itself, 10 after
                                    context_frames = self.get_context_frames(i, before_num = self.NUM_CONTEXT_BEFORE, after_num = self.NUM_CONTEXT_AFTER)

                                if self.args.classifier_function == 'linear':
                                    out_ = self.passive_clf(feat_vec)
                                    self.error_dict[i][(rule, task_phase, intervention)] = (
                                        F.softmax(out_, dim=0))
                            
                                elif self.args.classifier_function == 'vlm':
                                    p = self.base_prompt.replace("<insert rule index here>", str(idx))                                    
                                    # OpenCV frames are BGR numpy arrays; Molmo2 expects RGB PIL images
                                    image_list = [
                                        Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                                        for f in context_frames
                                    ]

                                    # Debug: 7 x 3 montage (7 columns, 3 rows) of the context frames
                                    make_montage(image_list, os.path.join(os.getcwd(), "tmp_montage.png"), (3, 7))

                                    out_ = self.passive_clf(prompt = p, image_list = image_list)
                                    print(out_)
                                    sys.exit()
                                    self.error_dict[i][(rule, task_phase, intervention)] = self.out_path   

                    for rule in violated:
                        cv2.imwrite(
                            os.path.join(self.args.save_dir, rule, f"{i}.png"), frame_bgr
                        )
                    if self.args.notification == "verbose":
                        self.mediapipe_obj.draw_skeleton(frame_bgr)
                        draw_rules(frame_bgr, self.sterile_field)
                    if self.args.mode == "realtime":
                        draw_sterile_field(frame_bgr, self.sterile_field, self.reader)

                    if writer is None:
                        writer = cv2.VideoWriter(
                            self.video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
                        )
                    writer.write(frame_bgr)

                    if self.args.mode == "realtime":
                        cv2.imshow("realtime", frame_bgr)
                        if cv2.waitKey(1) == ord("q"):
                            break
            finally:
                if num_detected:
                    scores = {k: 1 - v / num_detected for k, v in self.viol_frames.items()}
                    print(f"Frames with human detected: {num_detected}")
                    print(f"Number of frames violated for each rule is {self.viol_frames}")
                    print(f"Scores are {scores}")
                self.mediapipe_obj.close()
                if writer is not None:
                    writer.release()
                    print(f"Saved video to {self.video_path}")
                if raw_writer is not None:
                    raw_writer.release()
                if depth_writer is not None:
                    depth_writer.release()
                if self.args.save_pose and self.ts:
                    np.savez(
                        self.out_path,
                        timestamps_ms=np.array(self.ts),
                        landmarks_cam_m=np.stack(self.lms_cam),
                    )  # (T, 33, 3): X, Y, Z in meters
                    print(f"Saved pose to {self.out_path}")


if __name__ == "__main__":
    CLDCEvaluator(parse_args()).eval_cldc()
