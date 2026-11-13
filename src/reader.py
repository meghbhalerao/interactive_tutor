import cv2
import numpy as np
import pyrealsense2 as rs
import time
import os
import json
import src.domains.cldc.geometry as gm

class Reader():
    """
    mode is one of:
            'realtime': live data from the RealSense camera.
         'offline_mp4': data from recorded .mp4 videos.
    (The old 'offline_db3' mode lives in src/deprecated.py.)
    """

    def __init__(self, mode: str, path=None, color_resolution=[640, 480]) -> None:
        super().__init__()

        assert mode in ('realtime', 'offline_mp4'), f'Unknown Reader mode {mode!r}.'
        self.mode = mode
        if mode == 'realtime':
            self._init_realsense(color_resolution)
        elif mode == 'offline_mp4':
            assert path is not None, f'{mode} mode needs path.'
            self.mp4_rgb_path = os.path.join(path, 'realtime_raw.mp4')
            self.mp4_depth_path = os.path.join(path, 'depth_vals.npy')
            self._init_offline_mp4(os.path.join(path, 'realtime_depth_info.json'))

    def _init_offline_mp4(self, info_path):
        # Camera parameters saved by the realtime phase (the recorded frames are already undistorted).
        assert os.path.exists(info_path), f'offline_mp4 mode needs {info_path} (written by the realtime phase).'
        with open(info_path) as f:
            info = json.load(f)
        # stored swapped, like the live path: height_color is the frame width, width_color the frame height
        self.height_color, self.width_color = info['width'], info['height']
        self.depth_scale = info['depth_scale_m_per_count']
        self.new_camera_matrix = np.array([[info['fx'], 0, info['ppx']],
                                           [0, info['fy'], info['ppy']],
                                           [0, 0, 1]])

    def _init_realsense(self, color_resolution):
        # Define the frame shape
        self.height_color, self.width_color = color_resolution
        self.height_depth, self.width_depth = 640, 480
        self.can_align = (self.height_color, self.width_color) == (self.height_depth, self.width_depth)

        connected_cameras = Reader.find_connected_realsense_cameras()
        assert len(connected_cameras) > 0, 'There are no available RealSense cameras.'

        # Stream of RGB and depth data
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.height_color, self.width_color, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, self.height_depth, self.width_depth, rs.format.z16, 30)

        # Alignment
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.profile = self.pipeline.start(self.config)

        # Depth sensor and depth scale
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print("Depth Scale is: ", self.depth_scale)

        # Get camera intrinsic params
        intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                      [0, intrinsics.fy, intrinsics.ppy],
                                      [0, 0, 1]])
        self.dist_coeffs = np.array(intrinsics.coeffs)
        self.dist_model = intrinsics.model
        h, w = intrinsics.height, intrinsics.width
        self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h))

    @staticmethod
    def find_connected_realsense_cameras():
        """ Returns a list of all available RealSense cameras."""

        # Create a RealSense context
        ctx = rs.context()
        devices = ctx.query_devices()

        # List to store the connected RealSense cameras
        connected_cameras = []
        for dev in devices:
            if dev.supports(rs.camera_info.name):
                if "RealSense" in dev.get_info(rs.camera_info.name):
                    connected_cameras.append(dev)

        return connected_cameras

    def normalized_to_image_coordinates(self, x_norm, y_norm):
        # The image coordinate system x-right, y-down, (0,0)-top left corner
        x = x_norm * self.height_color
        y = y_norm * self.width_color
        return x, y
        
    def get_pixel_depth(self, depth_frame: np.ndarray, image_coordinate: gm.Point2D, kernel=5):
        """ Returns a z coordinate in camera coordinate system for a given pixel
            in image coordinate system.

        Args:
                 depth_frame: Numpy array containing depth values from Intel Realsense
                              sensors.
            image_coordinate: Tuple of x and y coordinate in image coordinate system.
                              Top-left corner is (0,0) and bottom-right corner is (h-1,w-1)
                 depth_scale: Scalar value that converts raw depth data to meters.
                      kernel: Define a neighboring range for pixel depth search.

        Returns:
            A single value that represents a z cooridnate in camera coordinate system.
        """

        def is_inside(i, j):
            dim0, dim1 = depth_frame.shape
            return (0 <= i < dim0) and (0 <= j < dim1)
        
        j_px, i_px = image_coordinate.unpack()

        non_zero_values = []
        for i in range(i_px-kernel, i_px+kernel+1):
            for j in range(j_px-kernel, j_px+kernel+1):
                if is_inside(i, j) and depth_frame[i][j] > 0:
                    non_zero_values.append(depth_frame[i][j])

        non_zero_values = np.array(non_zero_values, dtype=np.uint16)
        return np.median(non_zero_values)*self.depth_scale if non_zero_values.size > 0 else 0.0

    def get_frames(self):
        """ It returns a success flag as the first element of the tuple.
            Also, it returns an UNDISTORTED color frame (blue, green, 
            and red) ALIGNED with depth frame. Both frames are represented
            as NumPy arrays.
        """
            
        # Wait for a coherent pair of color and depth frames
        frames = self.pipeline.wait_for_frames()
        if self.can_align: 
            frames = self.align.process(frames)
        color_frame_rs = frames.get_color_frame()
        depth_frame_rs = frames.get_depth_frame()

        if not color_frame_rs or not depth_frame_rs:
            return False, None, None

        # Convert frames to numpy arrays
        color_frame = np.asanyarray(color_frame_rs.get_data())
        # color_frame = cv2.rotate(color_frame, cv2.ROTATE_90_CLOCKWISE)
        depth_frame = np.asanyarray(depth_frame_rs.get_data())
        # depth_frame = cv2.rotate(depth_frame, cv2.ROTATE_90_CLOCKWISE)

        # Undistort color frame
        color_frame = cv2.undistort(color_frame,
                                    self.camera_matrix,
                                    self.dist_coeffs,
                                    None,
                                    self.new_camera_matrix)

        return True, color_frame, depth_frame


    def frames(self):
        """ Yields (t_ms, bgr image, raw Z16 depth counts) for every frame of the source:
            forever for the live camera.
        """

        if self.mode == 'offline_mp4':
            depth_maps = np.load(self.mp4_depth_path, mmap_mode="r")

            cap = cv2.VideoCapture(self.mp4_rgb_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {self.mp4_rgb_path}")

            try:
                video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if video_frame_count != len(depth_maps):
                    print(
                        f"Warning: Video has {video_frame_count} frames, depth has"
                        f" {len(depth_maps)}."
                    )

                for depth_frame in depth_maps:
                    t_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))  # timestamp of the frame about to be read
                    ret, bgr_frame = cap.read()
                    if not ret:
                        break
                    # Keep BGR (like the realtime branch) and hand out an in-memory depth copy.
                    yield t_ms, bgr_frame, np.array(depth_frame)
            finally:
                cap.release()
            return

        # Reader aligns depth to color but only undistorts color, so depth is undistorted here with the same
        # camera matrices. Nearest-neighbour keeps every depth value a real measurement (no blending across edges).
        map_x = None  # built from the first depth frame's own size (width_color / height_color are swapped)
        while True:
            success, frame_bgr, depth_raw = self.get_frames()
            if success:
                if map_x is None:
                    size = (depth_raw.shape[1], depth_raw.shape[0])  # (width, height)
                    map_x, map_y = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix, size, cv2.CV_32FC1)
                depth_raw = cv2.remap(depth_raw, map_x, map_y, cv2.INTER_NEAREST)
                yield int(time.time() * 1000), frame_bgr, depth_raw

    def num_frames(self):
        """Frame count of the mp4 (offline_mp4 mode only)."""
        return int(cv2.VideoCapture(self.mp4_rgb_path).get(cv2.CAP_PROP_FRAME_COUNT))

    def __del__(self):
        if getattr(self, "mode", None) == 'realtime' and hasattr(self, "profile"):
            print('Stopping the RealSense pipeline.')
            self.pipeline.stop()
