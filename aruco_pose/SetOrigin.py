#!/usr/bin/env python3

import pyrealsense2 as rs
import cv2
import numpy as np
import cv2.aruco as aruco
import matplotlib.pyplot as plt

# This is to remove wayland error
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Camera matrix and distortion coefficients obtained from calibration
camera_matrix = np.array([[1366.5266, 0.0, 945.5471], [0.0, 1366.5305, 585.0791], [0.0, 0.0, 1.0]], dtype=float)
dist_coeffs = np.array([0.10600815, -0.43918719, -0.00832177, -0.00409687, 0.35950232])



# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)  # It should match with the resolution at which the camera was calibrated
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Get the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
parameters = cv2.aruco.DetectorParameters()

origin_marker_id_1 = 0  # Set the reference aruco ID, which will be set as the origin
transformation_matrix = np.zeros((4,4))

averaging = True
z_point  = np.array([0,0,0.5])  # Filter variable to discard faulty z axis
sample_size = 200
# Function to process frames and detect markers
def detect_markers():
    try:
        transformation_matrices = []
        j = 0
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # Detect markers
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            global averaging
            if ids is not None and averaging == True :
                color_image = aruco.drawDetectedMarkers(color_image, corners, ids) # Draw the detected markers in the image

                # Estimate pose of each marker
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners,0.1, camera_matrix, dist_coeffs)

                for i, marker_id in enumerate(ids):
                    if marker_id == origin_marker_id_1 : # If marker id is same as reference id 1  
                        origin_Translation = tvecs[i][0]
                        origin_Rotation = rvecs[i][0]
                        rotation_matrix,_ = cv2.Rodrigues(np.array(origin_Rotation))
                        new_transformation_matrix = np.eye(4)
                        new_transformation_matrix[:3, :3] = rotation_matrix
                        new_transformation_matrix[:3, 3] = origin_Translation
                        
                        
                        filter = np.dot(z_point, np.dot(rotation_matrix,z_point))

                        if(filter>0):
                            continue
                        elif filter<0 and averaging == True:
                            color_image = cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)
                            global transformation_matrix
                            transformation_matrix = np.add((1/sample_size) * new_transformation_matrix, transformation_matrix)
                            j = j + 1
                            print("Added new sample!",j)
                        
                        if j == sample_size:
                            averaging = False
                            print("Averaging done")
                        
            elif averaging == False:
                rotation_mat = transformation_matrix[:3, :3]

                # Use the Rodrigues function to convert the rotation matrix to a rotation vector
                rvec, _ = cv2.Rodrigues(rotation_mat)

                # Extract the translation vector
                tvec = transformation_matrix[:3, 3]
                color_image = cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

            # Display the resulting frame
            cv2.imshow('Frame', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("Transformation matrix:")
        print([list(i) for i in transformation_matrix])
        pipeline.stop()
        cv2.destroyAllWindows()

# Call the function to start detection
detect_markers()