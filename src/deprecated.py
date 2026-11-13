"""Deprecated: reading ROS 2 bag (.db3) recordings made with realsense-ros.

This code used to live in src/reader.py (mode 'offline_db3') and was moved here unchanged.
The methods below are those of the old Reader class and still rely on its attributes (self.db, self.intr, ...).
"""
import cv2
import numpy as np
import sqlite3
from types import SimpleNamespace
from rosbags.typesys import Stores, get_typestore

# ROS 2 bag (db3) recorded with realsense-ros
COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/"
DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/"
DEPTH_UNIT = 0.001  # m per Z16 count (Depth_Units option in the bag)
typestore = get_typestore(Stores.ROS2_HUMBLE)


class Db3Reader():
    """Old 'offline_db3' mode of Reader."""

    def _init_db3(self, db3_path):
        self.db = sqlite3.connect(f"file:{db3_path}?mode=ro", uri=True)

        self.intr = self._read_db3_intrinsics()
        self.height_color, self.width_color = self.intr.width, self.intr.height  # stored swapped, like the live path
        self.depth_scale = DEPTH_UNIT
        self.camera_matrix = self.new_camera_matrix = np.array([[self.intr.fx, 0, self.intr.ppx],
                                                                [0, self.intr.fy, self.intr.ppy],
                                                                [0, 0, 1]])
        self.dist_coeffs = np.zeros(5)

        # The first frame is what get_frames() serves (e.g. for clicking the sterile field)
        _, rgb, depth_m = next(self._read_db3_frames())
        self.color = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.depth = np.round(depth_m / DEPTH_UNIT).astype(np.uint16)  # raw Z16 counts, as the live path serves

    # --- from Reader.get_frames(): `if self.mode == 'offline_db3': return True, self.color.copy(), self.depth`

    def frames(self):
        """ Yields (t_ms, bgr image, raw Z16 depth counts) for every color frame of the db3 file."""
        for t_ms, rgb, depth_m in self._read_db3_frames():
            yield t_ms, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), np.round(depth_m / DEPTH_UNIT).astype(np.uint16)

    def _read_db3_intrinsics(self):
        """Color camera intrinsics from the bag's camera_info message."""
        raw = self.db.execute("select data from messages where topic_id=(select id from topics where name=?)", (COLOR_TOPIC + "camera_info",)).fetchone()[0]
        msg = typestore.deserialize_cdr(raw, "sensor_msgs/msg/CameraInfo")
        return SimpleNamespace(fx=msg.k[0], fy=msg.k[4], ppx=msg.k[2], ppy=msg.k[5], width=msg.width, height=msg.height)

    def _read_db3_depth_to_color(self):
        """(depth intrinsics, R, t) with p_color = R @ p_depth + t, from the bag's camera_info and tf metadata."""
        raw = self.db.execute("select data from messages where topic_id=(select id from topics where name=?)", (DEPTH_TOPIC + "camera_info",)).fetchone()[0]
        k = typestore.deserialize_cdr(raw, "sensor_msgs/msg/CameraInfo").k
        d_intr = SimpleNamespace(fx=k[0], fy=k[4], ppx=k[2], ppy=k[5])
        raw = self.db.execute("select data from messages where topic_id=(select id from topics where name='/device_0/sensor_1/Color_0/tf/ref_0')").fetchone()[0]
        fields = dict(f.split("=") for f in raw[8:].rstrip(b"\x00").decode().split(";"))
        R = np.array(fields["rotation"].split(","), dtype=float).reshape(3, 3).T  # librealsense stores rotation column-major
        t = np.array(fields["translation"].split(","), dtype=float)
        return d_intr, R, t

    def _align_db3_depth_to_color(self, depth_raw, d_intr, R, t):
        """Resample a raw Z16 depth image onto the color pixel grid -> (H, W) float depth in m (0 = no measurement)."""
        intr = self.intr
        v, u = np.nonzero(depth_raw)
        z = depth_raw[v, u] * DEPTH_UNIT
        p = np.stack([(u - d_intr.ppx) / d_intr.fx * z, (v - d_intr.ppy) / d_intr.fy * z, z], axis=1) @ R.T + t
        uc = np.round(p[:, 0] / p[:, 2] * intr.fx + intr.ppx).astype(int)
        vc = np.round(p[:, 1] / p[:, 2] * intr.fy + intr.ppy).astype(int)
        ok = (p[:, 2] > 0) & (uc >= 0) & (uc < intr.width) & (vc >= 0) & (vc < intr.height)
        out = np.full(intr.height * intr.width, np.inf)
        np.minimum.at(out, vc[ok] * intr.width + uc[ok], p[ok, 2])  # nearest surface wins where depth pixels collide
        out[np.isinf(out)] = 0
        return out.reshape(intr.height, intr.width)

    def _read_db3_frames(self):
        """Yield (t_ms, rgb image, aligned depth in m) for each color frame, paired with the nearest-in-time depth frame."""
        d_intr, R, t = self._read_db3_depth_to_color()
        d_rows = self.db.execute("select timestamp, data from messages where topic_id=(select id from topics where name=?) order by timestamp", (DEPTH_TOPIC + "data",)).fetchall()
        d_times = np.array([r[0] for r in d_rows])
        for t_ns, raw in self.db.execute("select timestamp, data from messages where topic_id=(select id from topics where name=?) order by timestamp", (COLOR_TOPIC + "data",)):
            color = typestore.deserialize_cdr(raw, "sensor_msgs/msg/Image")
            img = color.data.reshape(color.height, color.width, 3)  # rgb8
            j = int(np.argmin(np.abs(d_times - t_ns)))
            dm = typestore.deserialize_cdr(d_rows[j][1], "sensor_msgs/msg/Image")
            depth_raw = dm.data.view(np.uint16).reshape(dm.height, dm.width)
            yield int(t_ns // 1_000_000), img, self._align_db3_depth_to_color(depth_raw, d_intr, R, t)

    def __del__(self):
        if getattr(self, "mode", None) == 'offline_db3':
            self.db.close()
