#!/usr/bin/env python3

import pyrealsense2 as rs
import cv2
import numpy as np
import cv2.aruco as aruco
import matplotlib.pyplot as plt

# This is to remove wayland error
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
pipeline.start(config)

# Get the camera intrinsics
profile = pipeline.get_active_profile()
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
color_intrinsics = color_profile.get_intrinsics()

# Convert the intrinsics to numpy arrays
camera_matrix = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                          [0, color_intrinsics.fy, color_intrinsics.ppy],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([color_intrinsics.coeffs[0], color_intrinsics.coeffs[1], color_intrinsics.coeffs[2], color_intrinsics.coeffs[3]], dtype=np.float32)

print("Camera Matrix:")
print([list(i) for i in camera_matrix])
print("Distortion Coefficients")
print(dist_coeffs)
