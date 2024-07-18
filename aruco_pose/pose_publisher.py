#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import cv2
import numpy as np
import cv2.aruco as aruco
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import tf2_ros
from scipy.spatial.transform import Rotation as R

# This is to remove wayland error
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Camera matrix and distortion coefficients obtained from calibration
camera_matrix = np.array([[1366.5266, 0.0, 945.5471], [0.0, 1366.5305, 585.0791], [0.0, 0.0, 1.0]], dtype=float)
dist_coeffs = np.array([0.10600815, -0.43918719, -0.00832177, -0.00409687, 0.35950232])

# Get the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
parameters = cv2.aruco.DetectorParameters()

origin_transformation_matrix = np.array([[0.9899927575245974, -0.04548636918521066, 0.06550518336539876, -1.0266394771443363],
                                         [-0.046304058772116456, -0.9958656674151443, -0.06400782552137686, -0.1851852981129882],
                                         [0.06870039319380991, 0.06153835686330985, -0.9880363531716041, 2.495151906514195],
                                         [0.0, 0.0, 0.0, 1.0000000000000007]])

inverse_origin_transformation_matrix = np.linalg.inv(origin_transformation_matrix)
origin_rotation_mat = origin_transformation_matrix[:3, :3]
inverse_origin_rotation_mat = np.linalg.inv(origin_rotation_mat)
origin_rvec, _ = cv2.Rodrigues(origin_rotation_mat)
origin_tvec = origin_transformation_matrix[:3, 3]

z_point = np.array([0, 0, 1])  # Filter for faulty Z axis

kalman_filters = {}
frames_counter = 0

def initialize_kalman():
    """
    Initialize a Kalman filter for tracking the position of an ArUco marker.
    """
    tracker = cv2.KalmanFilter(6, 3)
    dt = 0.1

    tracker.transitionMatrix = np.array([[1, 0, 0, dt, 0, 0],
                                        [0, 1, 0, 0, dt, 0],
                                        [0, 0, 1, 0, 0, dt],
                                        [0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]], np.float32)
    
    tracker.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 0]], np.float32)
    
    tracker.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-4
    tracker.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-4
    tracker.errorCovPost = np.eye(6, dtype=np.float32)
    tracker.statePost = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

    return tracker

def kalman_correct(kalman, measurement):
    """
    Apply Kalman filter correction to the measurement.
    """
    prediction = kalman.predict()
    estimate = kalman.correct(np.array(measurement, dtype=np.float32))

    return prediction, estimate

class pose_publisher_node(Node):
    def __init__(self):
        super().__init__('pose_publisher')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

    def detect_markers(self):
        """
        Detect ArUco markers in the camera frame, estimate their poses, and publish the transforms.
        """
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

                # Detect markers
                corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

                if ids is not None:
                    # Estimate pose of each marker
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.1, camera_matrix, dist_coeffs)

                    for i, marker_id in enumerate(ids):
                        camera_Translation = tvecs[i][0]
                        camera_Rotation = rvecs[i][0]
                        rotation_matrix, _ = cv2.Rodrigues(np.array(camera_Rotation))

                        filter = np.dot(z_point, np.dot(rotation_matrix, z_point))
                        if filter > 0:
                            continue

                        position_camera = tvecs[i][0]
                        homogeneous_point = np.array([position_camera[0], position_camera[1], position_camera[2], 1.0])

                        net_transformation = np.eye(4)
                        net_transformation[:3, :3] = np.dot(inverse_origin_rotation_mat, rotation_matrix)
                        net_transformation[:3, 3] = np.dot(inverse_origin_transformation_matrix, homogeneous_point)[:3]

                        new_point = np.dot(inverse_origin_rotation_mat,(homogeneous_point[:3] - origin_tvec)) 

                        rotation_mat = net_transformation[:3, :3]
                        rvec, _ = cv2.Rodrigues(rotation_mat)
                        tvec = net_transformation[:3, 3]

                        self.publish_marker_transform(marker_id[0], tvec, rvec, z_lock = True)

                # Publish Camera transform
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = "camera"
                t.child_frame_id = "world"
                t.transform.translation.x = origin_tvec[0]
                t.transform.translation.y = origin_tvec[1]
                t.transform.translation.z = origin_tvec[2]

                
                q = R.from_matrix(origin_rotation_mat).as_quat()
                t.transform.rotation.x = q[0]
                t.transform.rotation.y = q[1]
                t.transform.rotation.z = q[2]
                t.transform.rotation.w = q[3]

                self.tf_broadcaster.sendTransform(t)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            self.get_logger().info                
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

    def publish_marker_transform(self, marker_id, tvec, rvec, z_lock):
        """
        Publish the transform of an ArUco marker.
        """

        global frames_counter, kalman_filters

        marker_ids = int(marker_id)
        if marker_ids not in kalman_filters:
            kalman_filters[marker_ids] = initialize_kalman()

        measurement = [tvec[0], tvec[1], tvec[2]]
        tracker = kalman_filters[marker_ids]

        if frames_counter == 0:
            frames_counter += 1
            tracker.processNoiseCov = np.eye(6, dtype=np.float32) * 1
        else:
            tracker.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2

        prediction, estimate = kalman_correct(tracker, measurement)
        prediction_values = prediction.ravel()

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = f"aruco_marker_{marker_ids}"
        t.transform.translation.x = float(estimate[0])
        t.transform.translation.y = float(estimate[1])
        t.transform.translation.z = float(estimate[2])
        
        if(z_lock == True):
            rot_mat, _ = cv2.Rodrigues(rvec)
            r = R.from_matrix(rot_mat)
            r0 = r.as_euler('zyx', degrees=True)
            r1 = R.from_euler('x',-r0[2], degrees=True)
            r2 = R.from_euler('y',-r0[1], degrees= True)
            new_r = r2*r1*r
            q = new_r.as_quat()
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
        else:
            # Convert rvec (Rodrigues angle) to quaternion
            rot_mat, _ = cv2.Rodrigues(rvec)
            q = R.from_matrix(rot_mat).as_quat()
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info(f"Marker ID: {marker_ids}, Estimated Position: x={estimate[0]:.2f}, y={estimate[1]:.2f}, z={estimate[2]:.2f}, Orientation: x={rvec[0][0]:.2f}, y={rvec[1][0]:.2f}, z={rvec[2][0]:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = pose_publisher_node()
    node.detect_markers()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
