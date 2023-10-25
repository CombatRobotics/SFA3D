import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image

import argparse
import sys
import os
import time
import warnings
import zipfile
from cv_bridge import CvBridge
warnings.filterwarnings("ignore", category=UserWarning)
import cv2
import torch
import numpy as np
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose, ObjectHypothesis
from norfair import Detection, Tracker, Video, draw_tracked_objects
from dynamic_radar_gridmap_msgs.msg import TrackedObject, TrackedObjectArray
from geometry_msgs.msg import PoseArray, Point, Pose

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.demo_dataset import Demo_KittiDataset
from models.model_utils import create_model
from utils.evaluation_utils import draw_predictions, convert_det_to_real_values
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.kitti_data_utils import Calibration
from utils.demo_utils import parse_demo_configs, do_detect, download_and_unzip, write_credit

class DetectionTracking(Node):
    def __init__(self, config):
        super().__init__("detection_node")
        self.configs = config 
        # Try to download the dataset for demonstration
        server_url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data'
        download_url = '{}/{}/{}.zip'.format(server_url, self.configs.foldername[:-5], self.configs.foldername)
        print(download_url)
        download_and_unzip(self.configs.dataset_dir, download_url)
        self.cvbridge = CvBridge()
        self.model = create_model(self.configs)
        print('\n\n' + '-*=' * 30 + '\n\n')
        assert os.path.isfile(self.configs.pretrained_path), "No file at {}".format(self.configs.pretrained_path)
        self.model.load_state_dict(torch.load(self.configs.pretrained_path, map_location='cpu'))
        print('Loaded weights from {}\n'.format(self.configs.pretrained_path))

        self.configs.device = torch.device('cpu' if self.configs.no_cuda else 'cuda:{}'.format(self.configs.gpu_idx))
        self.model = self.model.to(device=self.configs.device)
        self.model.eval()
        self.tracker = Tracker(distance_function= "euclidean"  , distance_threshold=200)

        self.out_cap = None
        self.demo_dataset = Demo_KittiDataset(self.configs)

        self.publisher_ = self.create_publisher(Image, 'out/image_raw', 10)
        self.markerarray_publisher_ = self.create_publisher(Detection3DArray, 'out/tracked_obj_arr', 10)
        self.marker_publisher_ = self.create_publisher(Detection3D, 'out/tracked_obj_mark', 10)
        self.radar_track_arr_pub = self.create_publisher(TrackedObjectArray, 'tracked_objects', 10)
        self.trajectory_pub = self.create_publisher(PoseArray, 'out/trajectory', 10)

        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.sample_idx = 0

    def timer_callback(self):
        
        
        with torch.no_grad():
            metadatas, bev_map, img_rgb = self.demo_dataset.load_bevmap_front(self.sample_idx)
            detections, bev_map, fps = do_detect(self.configs, self.model, bev_map, is_front=True)
           
            bev_map = (bev_map.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            bev_map = draw_predictions(bev_map, detections, self.configs.num_classes)

            # Rotate the bev_map
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

            img_path = metadatas['img_path'][0]
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            calib = Calibration(self.configs.calib_path)
            kitti_dets = convert_det_to_real_values(detections)
            # print(detections)
            # print(kitti_dets)
            if len(kitti_dets) > 0:
                kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
                img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)
            # _x, _y, _z = kitti_dets
            if len(kitti_dets) > 0:
                detected = []
                for det in kitti_dets:
                    # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                    _score, _x, _y, _z, _h, _w, _l, _yaw = det
                    

                    points = np.array(
                    [[_x, _y, _z]]
                    )
                    detected.append(Detection(points=points))
                tracked_objects = self.tracker.update(detected)

            ################## Default msg type ################
                objects_msg = Detection3DArray()
                objects_msg.header.stamp = self.get_clock().now().to_msg()
                objects_msg.header.frame_id = "base_link"
                for n,obj in enumerate(tracked_objects):
                    detection3d = Detection3D()
                    detection3d.header.stamp = self.get_clock().now().to_msg()
                    detection3d.header.frame_id = "base_link"
                    hypo = ObjectHypothesisWithPose()
                    hypo.hypothesis.class_id = str(obj.id)
                    hypo.hypothesis.score = 100.0
                    hypo.pose.pose.position.x = obj.estimate[0][2] #_z
                    hypo.pose.pose.position.y = obj.estimate[0][0] #_x
                    hypo.pose.pose.position.z = obj.estimate[0][1] #_y
                    # detection3d.results.clear()
                    detection3d.results.append(hypo)
                    detection3d.bbox.center.position.x = obj.estimate[0][2] #_z
                    detection3d.bbox.center.position.y = obj.estimate[0][0] #_x
                    detection3d.bbox.center.position.z = obj.estimate[0][1] #_y
                    detection3d.bbox.size.x = 2.5
                    detection3d.bbox.size.y = 1.5
                    detection3d.bbox.size.z = 1.1
                    detection3d.id = str(obj.id)
                    objects_msg.detections.append(detection3d)

                    self.marker_publisher_.publish(detection3d)
                self.markerarray_publisher_.publish(objects_msg)
            ################## ######################### ################

            tracked_arr = TrackedObjectArray() 
            tracked_arr.header.stamp = self.get_clock().now().to_msg()
            tracked_arr.header.frame_id = "base_link"
            pose_arr = PoseArray()
            pose_arr.header = tracked_arr.header
            
            for n,obj in enumerate(tracked_objects):
                print(obj.id, obj.estimate)
                tracked_obj = TrackedObject()
                tracked_obj.header.stamp = self.get_clock().now().to_msg()
                tracked_obj.header.frame_id = "base_link"
                tracked_obj.id = obj.id
                tracked_obj.object_class.data = "car"
                tracked_obj.position.position.x = obj.estimate[0][2] #_z
                tracked_obj.position.position.y = obj.estimate[0][0] #_x
                tracked_obj.position.position.z = obj.estimate[0][1] #_y
                tracked_obj.velocity.x = obj.estimate_velocity[0][2] #_z
                tracked_obj.velocity.y = obj.estimate_velocity[0][0] #_x
                tracked_obj.velocity.z = obj.estimate_velocity[0][1] #_y
                tracked_obj.size.x = 2.5
                tracked_obj.size.y = 1.5
                tracked_obj.size.z = 1.1
                num_steps = 10
                tracked_obj.trajectory.clear()

                new_position = np.array([obj.estimate[0][2], obj.estimate[0][0], obj.estimate[0][1]]) 
                for _ in range(num_steps):
                    new_position += np.array([obj.estimate_velocity[0][2], obj.estimate_velocity[0][0], obj.estimate_velocity[0][1]]) * 0.5
                    # print("new_position: ", new_position)
                    pose=Pose()
                    pose.position.x = new_position[0]
                    pose.position.y = new_position[1]
                    pose.position.z = new_position[2]
                    tracked_obj.trajectory.append(pose.position)
                    pose_arr.poses.append(pose)
                tracked_arr.tracked_objects.append(tracked_obj)
            self.radar_track_arr_pub.publish(tracked_arr)
            self.trajectory_pub.publish(pose_arr)




            out_img = merge_rgb_to_bev(img_bgr, bev_map, output_width=self.configs.output_width)
            ros_img_msg = self.cvbridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
            ros_img_msg.header.frame_id = "base_link"
            ros_img_msg.header.stamp = self.get_clock().now().to_msg()
            # print(ros_img_msg.header)
            self.publisher_.publish(ros_img_msg)
            # write_credit(out_img, (80, 210), text_author='Cre: github.com/maudzung', org_fps=(80, 250), fps=fps)
            if self.out_cap is None:
                out_cap_h, out_cap_w = out_img.shape[:2]
                # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                # out_path = os.path.join(self.configs.results_dir, '{}_front.avi'.format(self.configs.foldername))
                # print('Create video writer at {}'.format(out_path))

                # self.out_cap = cv2.VideoWriter(out_path, fourcc, 30, (out_cap_w, out_cap_h))

            # self.out_cap.write(out_img)
            # self.get_logger().info('Publishing: "%s"' % msg.data)
            self.sample_idx += 1
            if self.sample_idx >= len(self.demo_dataset):
                self.sample_idx = 0


def main(args=None):
    rclpy.init(args=args)
    conf = parse_demo_configs()
    minimal_publisher = DetectionTracking(conf)

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

