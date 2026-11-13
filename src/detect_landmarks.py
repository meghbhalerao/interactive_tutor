import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

MODEL_PATH = '/home/mb230/projects/adaptive_skill/assets/mediapipe_models/pose_landmarker.task'
INPUT_VIDEO = "/home/mb230/projects/adaptive_skill/data/example.mp4"
OUTPUT_VIDEO = "output_landmarks.mp4"


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  for pose_landmarks in pose_landmarks_list:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)

  return annotated_image


# Create a PoseLandmarker in VIDEO mode (tracks across frames).
options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO)
detector = vision.PoseLandmarker.create_from_options(options)

capture = cv2.VideoCapture(INPUT_VIDEO)
if not capture.isOpened():
  raise FileNotFoundError(f"Could not open video: {INPUT_VIDEO}")

fps = capture.get(cv2.CAP_PROP_FPS)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(
    OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

frame_idx = 0
while True:
  ok, frame_bgr = capture.read()
  if not ok:
    break

  frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
  image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
  timestamp_ms = int(frame_idx * 1000 / fps)

  detection_result = detector.detect_for_video(image, timestamp_ms)

  annotated_rgb = draw_landmarks_on_image(frame_rgb, detection_result)
  writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
  frame_idx += 1

capture.release()
writer.release()
detector.close()
print(f"Wrote {frame_idx} frames to {OUTPUT_VIDEO}")
