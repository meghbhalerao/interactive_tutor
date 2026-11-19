# code taken from the ASTRID paper
import cv2
import csv
import datetime
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pygame import mixer

import src.domains.cldc.geometry as gm
from src.reader import Reader
from src.utils.mediapipe_utils import MediaPipe, PoseLandmark

# rule shorthand -> index into sterile_field.is_contaminated output (shorthand is also the on-screen label)
rule_to_idx = {"hbt": 1, "ra": 2, "tb": 3, "oib": 4}


def draw_rules(frame_bgr, sterile_field):
    """Overlay 'human not found', or one line per rule (red = violated, green = ok) with the number it thresholds next to it. Call after sterile_field.is_contaminated."""
    if sterile_field.contaminations[0]:
        cv2.putText(frame_bgr, "human not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return
    for row, (label, idx) in enumerate(rule_to_idx.items()):
        violated = sterile_field.contaminations[idx]
        text = f"{'VIOLATED' if violated else 'ok'}: {label} = {sterile_field.rule_values[idx]:+.3f}"
        cv2.putText(frame_bgr, text, (10, 30 * (row + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if violated else (0, 160, 0), 2)


def draw_sterile_field(frame_bgr, sterile_field, reader):
    """Overlay the outer corners you clicked (red) and the computed one-inch inner border (black)."""
    for corners, color in ((sterile_field.out_corners_camera_coordinates, (0, 0, 255)), (sterile_field.in_corners_camera_coordinates, (0, 0, 0))):
        px = [gm.projectPoint(p, np.eye(3), np.zeros(3), reader.new_camera_matrix).unpack() for p in corners]
        for k in range(4):
            cv2.line(frame_bgr, px[k], px[(k + 1) % 4], color, 1)


class SterileField():

    def __init__(self, color='blue') -> None:
        """ Sterile Field Parameters

            out_corners_camera_coordinates - Stores 4 gm.Point3D objects which represent all four physical corners of the sterile field given in the camera coordinate system.
             in_corners_camera_coordinates - Stores 4 gm.Point3D objects which represent all four inner corners of the one-inch border of the sterile field given in the camera coordinate system. 
             plane - gm.Plane object that best mathematically represents the sterile field plane in 3D space.
            front_plane - gm.Plane object that best mathematically represents the front plane of the sterile field. This plane is used for detecting the "reaching over" type of contamination.
            color - Represents the color of the sterile field. It can be either 'red' or 'blue'.
            sterile_field_found - Flag which is set to True when the sterile filed is detected.
            contamination - Flag which represents the contamination of the sterile field.
        """

        self.out_corners_camera_coordinates = np.empty((4,), dtype=gm.Point3D)
        self.in_corners_camera_coordinates = np.empty((4,), dtype=gm.Point3D)
        self.plane = None
        self.front_plane = None

        # sterile field manual detection
        self.selected_corners = 0
        self.number_of_corners = 4

        # Define a range of color for the sterile field autonomous detection
        self.color = color
        if self.color == 'red':
            self.lower_bound = np.array([146, 100, 100], np.uint8)
            self.upper_bound = np.array([186, 255, 255], np.uint8)
        elif self.color == 'blue':
            self.lower_bound = np.array([50, 10, 10], np.uint8)
            self.upper_bound = np.array([140, 255, 255], np.uint8)
        else:
            raise Exception(f'The sterile field cannot be {color}.')

        # Flags
        self.sterile_field_found = False
        self.contaminations = np.full((5,), False)

        self.error_types = ['mistake', 'slip', 'lapse']
        
        # The number each rule thresholds (index = rule index in contaminations; 0 unused), NaN if not computable. For debugging.
        self.rule_values = np.full((5,), np.nan)

    def construct_sterile_field(self):
        """
        This method is used to construct a sterile field plane based on @out_corners_camera_coordinates
        list that is created either using the @detect_auto or @detect_manual method.
        """

        # Construct a plane - the points to be used to construct the plane are chosen arbitratily (at least that is what my understanding is)
        A = self.out_corners_camera_coordinates[0]
        B = self.out_corners_camera_coordinates[1]
        D = self.out_corners_camera_coordinates[3]
        self.plane = gm.Plane.from_three_points(A, B, D)

        # Project the 4th corner to the plane
        self.out_corners_camera_coordinates[2] = self.out_corners_camera_coordinates[2].orthogonal_projection(self.plane)
        C = self.out_corners_camera_coordinates[2]

        # Find inside corners of the one-inch border of the sterile field
        border_width_m = 0.0254
        self.number_of_corners = len(self.out_corners_camera_coordinates)
        for i in range(self.number_of_corners):
            # Create unit basis vectors
            previous = self.out_corners_camera_coordinates[(i-1) % self.number_of_corners]
            current = self.out_corners_camera_coordinates[i]
            following = self.out_corners_camera_coordinates[(i+1) % self.number_of_corners]
            base_vector_1 = gm.Vector3D.from_two_points(previous, current)
            base_vector_2 = gm.Vector3D.from_two_points(following, current)
            norm_1 = base_vector_1.norm()
            norm_2 = base_vector_2.norm()
            unit_base_vector_1 = base_vector_1 / norm_1
            unit_base_vector_2 = base_vector_2 / norm_2
            
            # Find a scaling factor
            base_vectors_cross_product_norm = gm.Vector3D.cross_product(base_vector_1, base_vector_2).norm()
            scaling_factor = border_width_m * norm_1 * norm_2 / base_vectors_cross_product_norm
            
            # Find a new point
            inside_point_vector = current.to_vector() + scaling_factor * (unit_base_vector_1 + unit_base_vector_2)
            self.in_corners_camera_coordinates[i] = inside_point_vector.to_point()

        # Construct a front plane
        corners_sorted = sorted(self.out_corners_camera_coordinates, key=lambda corner: corner.z)
        A = corners_sorted[0]
        B = corners_sorted[1]
        C = B.to_vector() + self.plane.normal_vector()
        self.front_plane = gm.Plane.from_three_points(A, B, C.to_point())

        # Flags
        self.sterile_field_found = True

    def display_sterile_field(self, color_frame: np.ndarray, reader: Reader, depth_vis: np.ndarray = None):
        """
        This method is used to print, plot and display adequate information
        regarding the sterile field.

        Args:
            color_frame (np.ndarray): one-inch border will be displayed on top
                of the color_frame
            reader (Reader): An instance object of the class Reader which is in
                charge of capturing data from Intel RealSense cameras.
            depth_vis (np.ndarray): optional colorized depth image; the
                one-inch border is drawn on it as well.
        """

        # Prints
        print(self.plane)
        print(self.front_plane)
        print(self.in_corners_camera_coordinates)
        print(self.out_corners_camera_coordinates)

        for i in range(4):
            x1, y1, z1 = self.out_corners_camera_coordinates[i].unpack()
            x2, y2, z2 = self.in_corners_camera_coordinates[i].unpack()
            print(x2-x1, y2-y1, z2-z1)

        # 3D plots
        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('3D Sterile Field Border Projection')
        x_in = [corner.x for corner in self.in_corners_camera_coordinates]
        y_in = [corner.y for corner in self.in_corners_camera_coordinates]
        z_in = [corner.z for corner in self.in_corners_camera_coordinates]
        ax.scatter3D(x_in, y_in, z_in, color='blue')
        x_out = [corner.x for corner in self.out_corners_camera_coordinates]
        y_out = [corner.y for corner in self.out_corners_camera_coordinates]
        z_out = [corner.z for corner in self.out_corners_camera_coordinates]
        ax.scatter3D(x_out, y_out, z_out, color='red')
        plt.show()

        # Border display
        for i in range(self.number_of_corners):
            # the frames are already undistorted with new_camera_matrix, so project with it and no distortion
            curr_pixel = gm.projectPoint(self.in_corners_camera_coordinates[i], np.eye(3), np.zeros((3)), reader.new_camera_matrix)
            next_pixel = gm.projectPoint(self.in_corners_camera_coordinates[(i+1) % self.number_of_corners], np.eye(3), np.zeros((3)), reader.new_camera_matrix)
            cv2.line(color_frame, curr_pixel.unpack(), next_pixel.unpack(), (0, 0, 0))
            if depth_vis is not None:
                cv2.line(depth_vis, curr_pixel.unpack(), next_pixel.unpack(), (0, 0, 0))

    def detect_manual(self, reader_mode, select_mode, path=None, reader=None):
        """ This function gathers all information required to mathematically mode the sterile field. That includes 3D positions of all 4 outer corners in
            the camera coordinate system, inner corners, and both plane and front plane
            of the sterile field.

            reader - optional frame source with the same interface as Reader (e.g. one that replays a .db3 bag).
                     If None, a Reader is created from reader_mode and name.
        """

        clicked_pixels = []

        def mouse_callback(event, x, y, flags, param):
            image = param
            height, width, nmb_of_channels = image.shape

            if event ==  cv2.EVENT_LBUTTONUP:
                if self.selected_corners < self.number_of_corners:
                    if 0 <= x < width and 0 <= y < height:
                        print(f"Mouse released at position: ")
                        print(f"\tx = {x}")
                        print(f"\ty = {y}")
                        
                        # Find 3D coordinate of corners
                        image_coordinate = gm.Point2D([x, y])
                        z_c = reader.get_pixel_depth(depth_frame, image_coordinate, kernel=0)
                        if z_c == 0:
                            print("Depth is not found for this corner. Try again!")
                            return

                        corner_3D = gm.unprojectPoint(image_coordinate, z_c, reader.new_camera_matrix)
                        self.out_corners_camera_coordinates[self.selected_corners] = corner_3D
                        self.selected_corners += 1

                        # Show the selected corners and the order they were picked in, on both the RGB and depth images
                        clicked_pixels.append((x, y))
                        print(f"\tdepth = {z_c:.3f} m")
                        for panel in (image, depth_vis):
                            cv2.circle(panel, (x, y), 2, (0, 0, 255), -1)
                            cv2.putText(panel, f'{len(clicked_pixels)}', (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.putText(panel, f'{z_c:.3f}m', (x - 40, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            if len(clicked_pixels) > 1:
                                cv2.line(panel, clicked_pixels[-2], clicked_pixels[-1], (0, 0, 255), 1)
                            if len(clicked_pixels) == self.number_of_corners:
                                cv2.line(panel, clicked_pixels[-1], clicked_pixels[0], (0, 0, 255), 1)
                        cv2.imshow('Sterile Field Detection', np.hstack((image, depth_vis)))
                    else:
                        print("Mouse released outside the image window.")
                else:
                    print(f"You have already selected all {self.number_of_corners} corners.")

        # Instantiate Reader
        if reader is None:
            reader = Reader(mode=reader_mode, path=path)

        # Intel RealSense image stabilization time
        _IMAGE_STABILIZATION = 100
        cnt = 0

        # Take the first frame of the sterile field
        success = False
        color_frame = None
        depth_frame = None
        while cnt <= _IMAGE_STABILIZATION:
            success, color_frame, depth_frame = reader.get_frames()
            if success:
                cnt += 1

        # Mirror the frame
        # color_frame = cv2.flip(color_frame, 1)
        # depth_frame = cv2.flip(depth_frame, 1)

        # Depth image for reference only (RealSense convention: blue = near, red = far, black = no depth)
        valid_depth = depth_frame > 0
        depth_vis = np.zeros((*depth_frame.shape, 3), dtype=np.uint8)
        if valid_depth.any():
            near, far = depth_frame[valid_depth].min(), depth_frame[valid_depth].max()
            depth_scaled = ((depth_frame.astype(np.float32) - near) / max(far - near, 1) * 255).clip(0, 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
            depth_vis[~valid_depth] = 0

        # Create a Display Window
        cv2.namedWindow('Sterile Field Detection', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Sterile Field Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        if select_mode != 'rgb':
            raise ValueError(f"Sterile selection mode {select_mode} is not defined! Only rgb is supported (depth is shown beside it for reference)")

        # Clicks are only accepted on the RGB image (left half)
        cv2.setMouseCallback('Sterile Field Detection', mouse_callback, color_frame)

        cv2.putText(color_frame, f'Select corners clockwise starting top left', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))

        cv2.imshow('Sterile Field Detection', np.hstack((color_frame, depth_vis)))

        # Manually select 4 corners
        while self.selected_corners < self.number_of_corners:
            cv2.waitKey(1)

        # Let the window repaint (shows the 4th corner) before the blocking 3D plot opens
        cv2.waitKey(1)

        # Construct sterile field plane
        self.construct_sterile_field()

        # Display
        self.display_sterile_field(color_frame, reader, depth_vis)
        cv2.imshow('Sterile Field Detection', np.hstack((color_frame, depth_vis)))

        # Close cv2 window and reader
        del reader
        cv2.waitKey()
        cv2.destroyWindow('Sterile Field Detection')
        
    def detect_auto(self, reader: Reader):
        """ This function gathers all information required to mathematically model
            the sterile field. That includes 3D positions of all 4 outer corners in
            the camera coordinate system, inner corners, and both plane and front plane
            of the sterile field.

        Args:
            reader - An instance object of the class Reader which is in charge of
                     capturing data from Intel RealSense cameras.
        """

        # Create a Display Window
        cv2.namedWindow('Sterile Field Detection', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Sterile Field Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while not self.sterile_field_found:

            success, color_frame, depth_frame = reader.get_frames()
            if not success:
                continue

            # Define a mask in HSV color space
            color_frame_hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(color_frame_hsv, self.lower_bound, self.upper_bound)
            # mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8))
            mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8))

            # Find the best available contour
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour_approx = None
            current_max_area = 0
            min_allowed_area = 1000
            for contour in contours:

                # Determine the shape of the contour
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                if len(approx) != 4:
                    print("it is not 4")
                    continue

                # Determine the area of contour
                area = cv2.contourArea(contour)
                if (area > min_allowed_area) and (area > current_max_area):
                    best_contour_approx = approx
                    current_max_area = area

            if best_contour_approx is not None:
                all_corners_depth_found = True
                for i in range(len(best_contour_approx)):

                    # Find 3D coordinate of corners
                    image_coordinate = gm.Point2D(best_contour_approx[i][0])
                    z_c = reader.get_pixel_depth(depth_frame, image_coordinate)
                    if z_c == 0:
                        all_corners_depth_found = False
                        break

                    corner_3D = gm.unprojectPoint(image_coordinate, z_c, reader.new_camera_matrix)
                    self.out_corners_camera_coordinates[i] = corner_3D

                    # Index corners
                    cv2.putText(color_frame, f'{i}', best_contour_approx[i][0],  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
                    
                    # Draw a line connecting two corners
                    cv2.line(color_frame, best_contour_approx[i][0], best_contour_approx[(i+1)%len(best_contour_approx)][0], (0, 0, 0))
                
                if all_corners_depth_found:

                    # Construct sterile field plane
                    self.construct_sterile_field()

                    # Display
                    self.display_sterile_field(color_frame, reader)

                else:
                    print("Depth is not found for all corners.")
            else:
                print("Nothing adequate was found.")

            # OpenCV display frame
            images = np.hstack((color_frame, np.stack((mask, mask, mask), axis=-1)))
            cv2.imshow('Sterile Field Detection', images)
            if cv2.waitKey(1) == ord('q'):
                break
            
        del reader
        cv2.waitKey()
        cv2.destroyWindow('Sterile Field Detection')

    def is_contaminated(self, human: MediaPipe):

        def is_hand_below_waistline():
            """ 
            Returns:
                It returns True if any of the human's hands are below
                the waistline. Otherwise, it returns False.
            """
            
            # Define 3D points
            left_hip = human.pose_keypoints_camera_coordinates[PoseLandmark.LEFT_HIP]
            left_wrist = human.pose_keypoints_camera_coordinates[PoseLandmark.LEFT_WRIST]
            right_wrist = human.pose_keypoints_camera_coordinates[PoseLandmark.RIGHT_WRIST]

            if left_hip is None or left_wrist is None or right_wrist is None:
                # return True
                return False

            # Calculate the contamination
            sterile_field_normal_vector = self.plane.normal_vector()

            waistline_plane = gm.Plane.from_normal_vector_and_point(sterile_field_normal_vector, left_hip)
            # the camera is above the table, so "up" is the camera's side of the plane
            up_side = np.sign(waistline_plane.signed_distance(gm.Point3D([0, 0, 0])))
            # lowest wrist height above the waistline (m); violated when < 0
            self.rule_values[1] = min(waistline_plane.signed_distance(left_wrist) * up_side, waistline_plane.signed_distance(right_wrist) * up_side)
            return self.rule_values[1] < 0
            
        def is_hand_reaching_over():
            """ 
            Returns:
                It returns True if the human is reaching over the sterile
                field. Otherwise, it returns False.
            """
            
            # Define 3D points
            camera_origin = gm.Point3D([0, 0, 0])
            left_wrist = human.pose_keypoints_camera_coordinates[PoseLandmark.LEFT_WRIST]
            right_wrist = human.pose_keypoints_camera_coordinates[PoseLandmark.RIGHT_WRIST]

            if left_wrist is None and right_wrist is None:
                # print("Left wrist and right wrist cannot be detected.")
                return False

            # Calculate the contamination
            contaminated = False
            camera_side = np.sign(self.front_plane.signed_distance(camera_origin))
            wrist_distances = []  # (m) how far each wrist is on the camera's side of the front plane; violated when > 0
            if left_wrist is not None:
                contaminated |= self.front_plane.on_same_side(camera_origin, left_wrist)
                wrist_distances.append(self.front_plane.signed_distance(left_wrist) * camera_side)
            if right_wrist is not None:
                contaminated |= self.front_plane.on_same_side(camera_origin, right_wrist)
                wrist_distances.append(self.front_plane.signed_distance(right_wrist) * camera_side)
            self.rule_values[2] = max(wrist_distances)

            return contaminated
        
        def is_back_turned_on_sterile_field():
                """ 
                Returns:
                    It returns True if the human is not facing towards the
                    sterile field. Otherwise, it returns False.
                """

                # Define 3D points
                left_shoulder = human.pose_keypoints_camera_coordinates[PoseLandmark.LEFT_SHOULDER]
                right_shoulder = human.pose_keypoints_camera_coordinates[PoseLandmark.RIGHT_SHOULDER]

                if left_shoulder is None or right_shoulder is None:
                    return True

                # Define a hip vector
                left_shoulder_to_right_shoulder = gm.Vector3D.from_two_points(left_shoulder, right_shoulder)

                # Calculate the contamination
                self.rule_values[3] = left_shoulder_to_right_shoulder.x  # (m) violated when < 0
                contaminated = left_shoulder_to_right_shoulder.x < 0

                return contaminated
        
        def one_inch_border_contamination():
            """ 
            Returns:
                It returns True if hands are touching any part of the
                one-inch border of the sterile field. Otherwise, it 
                returns False.
            """
            
            def is_inside_polygon(P: gm.Point3D, polygon: np.ndarray[gm.Point3D]):
                
                orientations = []
                number_of_corners = len(polygon)
                for i in range(number_of_corners):
                    A = polygon[i]
                    B = polygon[(i+1)%number_of_corners]
                    AB = gm.Vector3D.from_two_points(B, A)
                    AP = gm.Vector3D.from_two_points(P, A)
                    cross_product_vector = gm.Vector3D.cross_product(AB, AP)
                    orientations.append(cross_product_vector.y > 0)

                return all(o == orientations[0] for o in orientations)


            MAX_TOUCH_DISTANCE_FROM_STERILE_FIELD = 0.025 * 3

            # Touching test
            contaminated = False
            for hand in human.hands_keypoints_camera_coordinates:
                for hand_landmark_3D in hand:
                    if hand_landmark_3D is not None:
                        # Touching
                        dist_from_sterile_field = hand_landmark_3D.get_distance_from_plane(self.plane)
                        # closest any hand landmark gets to the field plane (m); violated when < MAX_TOUCH_DISTANCE_FROM_STERILE_FIELD and over the border
                        self.rule_values[4] = np.fmin(self.rule_values[4], dist_from_sterile_field)
                        if dist_from_sterile_field < MAX_TOUCH_DISTANCE_FROM_STERILE_FIELD:
                            # Check which part of the sterile field is
                            projected_hand_landmark_3D = hand_landmark_3D.orthogonal_projection(self.plane)
                            if is_inside_polygon(projected_hand_landmark_3D, self.out_corners_camera_coordinates) and \
                                not is_inside_polygon(projected_hand_landmark_3D, self.in_corners_camera_coordinates):
                                    contaminated = True
                                
            return contaminated

        if not self.sterile_field_found:
            exit('The sterile field has not been found yet.')
        
        self.contaminations = np.full((5,), False)
        self.rule_values = np.full((5,), np.nan)

        # index 0 = no human found; the other rules cannot be evaluated without one
        if not human.human_found:
            self.contaminations[0] = True
            print('Human cannot be found.')
            return self.contaminations

        self.contaminations[1] = is_hand_below_waistline()
        self.contaminations[2] = is_hand_reaching_over()
        self.contaminations[3] = is_back_turned_on_sterile_field()

        if human.human_hands_found:
            self.contaminations[4] = one_inch_border_contamination()
        else:
            print('Human\'s hand cannot be found.')

        # self.contamination_dict = {"hand_or_human_found": self.contaminations[0], 
        # "hand_below_waist": self.contaminations[1],
        # "hand_reaching_over": self.contaminations[2],
        # "turned_away": self.contaminations[3],
        # "touching_one_inch": self.contaminations[4]}
        
        return self.contaminations
    
    def alarm_contamination(self, color_frame):

        if not np.any(self.contaminations):
            print('The sterile field is clean.')

        # Display Warnings on the image
        cnt = 0
        contaminations_list = ['Human must be present.',
                               'Your hands must be above the waistline.',
                               'You must not reach over the sterile field.',
                               'You must not turn your back on the sterile field.',
                               'You must not touch one inch border of the sterile field.']
        for i, contamination in enumerate(self.contaminations):
            if i>=1 and contamination:
                cnt += 1
                cv2.putText(color_frame, f'WARNING: {contaminations_list[i]}', (10, 30*cnt), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)




# def record_data_points():
#     with open("3D_points.csv", "a", newline="") as csvfile:
#         current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         fieldnames = ["timestamp", "left_shoulder_x", "left_shoulder_y", "left_shoulder_z", "right_shoulder_x", "right_shoulder_y", "right_shoulder_z"]
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames) 

#         left_shoulder = human.pose_keypoints_camera_coordinates[PoseLandmark.LEFT_SHOULDER]
#         right_shoulder = human.pose_keypoints_camera_coordinates[PoseLandmark.RIGHT_SHOULDER]

#         data = {
#             "timestamp": current_time,
#             "left_shoulder_x": left_shoulder.unpack()[0],
#             "left_shoulder_y": left_shoulder.unpack()[1], 
#             "left_shoulder_z": left_shoulder.unpack()[2],
#             "right_shoulder_x": right_shoulder.unpack()[0],
#             "right_shoulder_y": right_shoulder.unpack()[1],
#             "right_shoulder_z": right_shoulder.unpack()[2]
#         }
#         writer.writerow(data)