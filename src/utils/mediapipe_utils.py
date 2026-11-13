# File is originally from the ASTRID paper 

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils

from enum import Enum

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import src.domains.cldc.geometry as gm
from src.reader import Reader

class PoseLandmark():
    """The 33 pose landmarks."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

class MediaPipe():

    def __init__(self, pose_model_path: str, hand_model_path: str, min_detection_confidence=0.5, min_tracking_confidence=0.5) -> None:
        """ Human Parameters

                       pose_model_path - Path to the MediaPipe Pose Landmarker .task file.
                       hand_model_path - Path to the MediaPipe Hand Landmarker .task file.
             image coordinate keypoint - An array of Point3D objects which represent the landmarks
                                         positions in pixel coordinate system. (0, 0) is top-left 
                                         corner. Direction of x axis is from left to right, while 
                                         direction of y axis is from top to bottom.
            camera coordinate keypoint - An array of Point3D objects which represent the landmarks
                                         positions in camera coordinate system.
                           human_found - Flag which is set to True when the Mediapipe model detects
                                         human pose. Flag is reseted every frame.
                     human_hands_found - Flag which is set to True when the Mediapipe model detects
                                         human hands. Flag is reseted every frame.
        """

        # Define models (MediaPipe Tasks API, VIDEO mode: every call needs a strictly increasing timestamp)
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=pose_model_path),
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence))
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2))

        # Skeleton connections as (start, end) landmark index pairs
        self.pose_connections = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
        self.hand_connections = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]

        # Define keypoints for Human Pose Estimation
        self.pose_results = None
        self.pose_number_of_keypoints = 33
        self.pose_keypoints_normalized = np.empty((self.pose_number_of_keypoints, 3))
        self.pose_keypoints_image_coordinates = np.empty((self.pose_number_of_keypoints, ), dtype=gm.Point2D)
        self.pose_keypoints_camera_coordinates = np.empty((self.pose_number_of_keypoints, ), dtype=gm.Point3D)

        # Define keypoints for Hand Tracking
        self.hands_results = None
        self.hands_number_of_keypoints = 21
        self.hands_keypoints_normalized = np.empty((2, self.hands_number_of_keypoints, 3))
        self.hands_keypoints_image_coordinates = np.empty((2, self.hands_number_of_keypoints), dtype=gm.Point2D)
        self.hands_keypoints_camera_coordinates = np.empty((2, self.hands_number_of_keypoints), dtype=gm.Point3D)

        # Flags
        self.human_found = False
        self.human_hands_found = False

    def estimate_pose(self, color_frame, depth_frame, reader: Reader, timestamp_ms):
        """ Human Pose Estimation on a single frame.

        Args:
            color_frame - NumPy array which represents the image on which the
                          inference should be run.
            depth_frame - NumPy array which represents the z-coordinate (depth)
                          of each pixel in camera coordinate system.
                 reader - An instance object of the class Reader which is in charge of
                          capturing data from Intel RealSense cameras.
           timestamp_ms - Frame timestamp in ms; must be strictly increasing across calls.

        Returns:
        """
        
        # Convert from MediaPipe structure to Python touple
        def unpack_landmark(landmark):
            x = landmark.x
            y = landmark.y
            z = landmark.z
            visibility = landmark.visibility
            return x, y, z, visibility
    
        _VISIBILITY_THRESHOLD = 0.5
        _PRESENCE_THRESHOLD = 0.5
        frame_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        self.human_found = False

        # Reset the coordinates
        self.pose_keypoints_normalized = np.empty((self.pose_number_of_keypoints, 3))
        self.pose_keypoints_image_coordinates = np.empty((self.pose_number_of_keypoints, ), dtype=gm.Point2D)
        self.pose_keypoints_camera_coordinates = np.empty((self.pose_number_of_keypoints, ), dtype=gm.Point3D)
        
        # Estimate Human Pose
        self.pose_results = self.pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb), timestamp_ms)
        if self.pose_results.pose_landmarks:
            for idx, landmark in enumerate(self.pose_results.pose_landmarks[0]):
                
                # Update normalized keypoint
                x, y, z, visibility = unpack_landmark(landmark)
                self.pose_keypoints_normalized[idx] = np.array([x, y, z])
                
                # Not enough visibility to be considered an accurate landmark
                if ((landmark.visibility is not None and landmark.visibility < _VISIBILITY_THRESHOLD) or
                    (landmark.presence is not None and landmark.presence < _PRESENCE_THRESHOLD)):
                    continue

                # Update image coordinate keypoint
                x, y = reader.normalized_to_image_coordinates(x, y)
                self.pose_keypoints_image_coordinates[idx] = gm.Point2D([np.int32(x), np.int32(y)])

                # Update a depth for each keypoint
                z_c = reader.get_pixel_depth(depth_frame, self.pose_keypoints_image_coordinates[idx])
                if z_c > 0:
                    landmark_3D = gm.unprojectPoint(self.pose_keypoints_image_coordinates[idx], z_c, reader.new_camera_matrix)
                    self.pose_keypoints_camera_coordinates[idx] = landmark_3D

            # Update that the human was detected
            self.human_found = True

    def estimate_hands(self, color_frame, depth_frame, reader: Reader, timestamp_ms):
        """ Hand Pose Estimation on a single frame.

        Args:
            color_frame - NumPy array which represents the image on which the
                          inference should be run.
            depth_frame - NumPy array which represents the z-coordinate (depth)
                          of each pixel in camera coordinate system.
                 reader - An instance object of the class Reader which is in charge of
                          capturing data from Intel RealSense cameras.
           timestamp_ms - Frame timestamp in ms; must be strictly increasing across calls.

        Returns:
        """
        
        # Convert from MediaPipe structure to Python touple
        def unpack_landmark(landmark):
            x = landmark.x
            y = landmark.y
            z = landmark.z
            return x, y, z
        
        _VISIBILITY_THRESHOLD = 0.5
        _PRESENCE_THRESHOLD = 0.5
        frame_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        self.human_hands_found = False
        
        # Reset the coordinates
        self.hands_keypoints_normalized = np.empty((2, self.hands_number_of_keypoints, 3))
        self.hands_keypoints_image_coordinates = np.empty((2, self.hands_number_of_keypoints), dtype=gm.Point2D)
        self.hands_keypoints_camera_coordinates = np.empty((2, self.hands_number_of_keypoints), dtype=gm.Point3D)

        # Hand Tracking
        self.hands_results = self.hands.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb), timestamp_ms)
        if self.hands_results.hand_landmarks and len(self.hands_results.hand_landmarks)<=2:
            for hand_idx, hand_landmarks in enumerate(self.hands_results.hand_landmarks):
                for idx, landmark in enumerate(hand_landmarks):
                    
                    # Update normalized keypoint
                    x, y, z = unpack_landmark(landmark)
                    self.hands_keypoints_normalized[hand_idx][idx] = np.array([x, y, z])

                    # Not enough visibility to be considered an accurate landmark
                    if ((landmark.visibility is not None and landmark.visibility < _VISIBILITY_THRESHOLD) or
                        (landmark.presence is not None and landmark.presence < _PRESENCE_THRESHOLD)):
                        continue

                    # Update image coordinate keypoint
                    x, y = reader.normalized_to_image_coordinates(x, y)
                    self.hands_keypoints_image_coordinates[hand_idx][idx] = gm.Point2D([np.int32(x), np.int32(y)])

                    # Update a depth for each keypoint
                    z_c = reader.get_pixel_depth(depth_frame, self.hands_keypoints_image_coordinates[hand_idx][idx])
                    if z_c > 0:
                        landmark_3D = gm.unprojectPoint(self.hands_keypoints_image_coordinates[hand_idx][idx], z_c, reader.new_camera_matrix)
                        self.hands_keypoints_camera_coordinates[hand_idx][idx] = landmark_3D

            # Update that the human's hands were detected
            self.human_hands_found = True

    def draw(self, frame):
        """ Draw a skeleton for both human and hand pose estimation on a given frame. """

        # Draw a Human Pose Estimation
        if self.human_found:
            drawing_utils.draw_landmarks(frame, self.pose_results.pose_landmarks[0], vision.PoseLandmarksConnections.POSE_LANDMARKS)

        # Draw a Hand Tracking
        if self.human_hands_found:
            for hand_landmarks in self.hands_results.hand_landmarks:
                drawing_utils.draw_landmarks(frame, hand_landmarks, vision.HandLandmarksConnections.HAND_CONNECTIONS)

        return frame

    def close(self):
        """ Release both MediaPipe landmarkers. """
        self.pose.close()
        self.hands.close()

    def plot_3D(self):
        """ Create a 3D Matplotlib plot of both human and hand pose estimation. """
        
        # Define a 3D plot
        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('3D Human Pose Estimation')

        # Plot Human Pose Estimation
        if self.human_found:

            # Scatter 3D keypoints
            x, y, z = [], [], []
            for keypoint in self.pose_keypoints_camera_coordinates:
                if keypoint is not None:
                    x.append(keypoint.x)
                    y.append(keypoint.y)
                    z.append(keypoint.z)
            ax.scatter3D(x, y, z, color='red')

            # Display connections between keypoints
            for connection in self.pose_connections:
                start_idx = connection[0]
                end_idx = connection[1]
                if not (0 <= start_idx < self.pose_number_of_keypoints and 0 <= end_idx < self.pose_number_of_keypoints):
                    raise ValueError(f'Landmark index is out of range. Invalid connection '
                                    f'from landmark #{start_idx} to landmark #{end_idx}.')
                
                start_point = self.pose_keypoints_camera_coordinates[start_idx]
                end_point = self.pose_keypoints_camera_coordinates[end_idx]
                if start_point is not None and end_point is not None:
                    ax.plot3D(
                        xs=[start_point.x, end_point.x],
                        ys=[start_point.y, end_point.y],
                        zs=[start_point.z, end_point.z],
                        color='blue',
                        linewidth=2)
                
        # Plot Hand Tracking
        if self.human_hands_found:

            number_of_found_hands = range(len(self.hands_results.hand_landmarks))
            for hand_idx in number_of_found_hands:

                # Scatter 3D keypoints
                x, y, z = [], [], []
                for keypoint in self.hands_keypoints_camera_coordinates[hand_idx]:
                    if keypoint is not None:
                        x.append(keypoint.x)
                        y.append(keypoint.y)
                        z.append(keypoint.z)
                ax.scatter3D(x, y, z, color='red')

                # Display connections between keypoints
                for connection in self.hand_connections:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    if not (0 <= start_idx < self.pose_number_of_keypoints and 0 <= end_idx < self.pose_number_of_keypoints):
                        raise ValueError(f'Landmark index is out of range. Invalid connection '
                                        f'from landmark #{start_idx} to landmark #{end_idx}.')
                    
                    start_point = self.hands_keypoints_camera_coordinates[hand_idx][start_idx]
                    end_point = self.hands_keypoints_camera_coordinates[hand_idx][end_idx]
                    if start_point is not None and end_point is not None:
                        ax.plot3D(
                            xs=[start_point.x, end_point.x],
                            ys=[start_point.y, end_point.y],
                            zs=[start_point.z, end_point.z],
                            color='blue',
                            linewidth=2)

        plt.show()

    def reset_flags(self):

        # Reset all flag variables
        self.human_found = False
        self.human_hands_found = False


