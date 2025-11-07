#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import os
import struct
import random
import yaml

class MultiObjectPublisher(Node):
    def __init__(self, folder_path, yaml_file, topic='/scene_colored'):
        super().__init__('multi_object_publisher')
        self.folder_path = folder_path

        # ROS2 parameters
        self.declare_parameter('publish_rate', 1.0)  # Hz
        self.declare_parameter('frame_id', 'world')
        publish_rate = self.get_parameter('publish_rate').value
        self.frame_id = self.get_parameter('frame_id').value

        self.get_logger().info(f'Publishing point clouds and markers')

        # Publishers
        self.pc_pub = self.create_publisher(PointCloud2, topic, 1)
        self.marker_pub = self.create_publisher(MarkerArray, topic+'_markers', 1)

        self.timer = self.create_timer(1.0 / publish_rate, self.publish_pointcloud)

        # Preload PLY files
        self.ply_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.ply')])

        # Load YAML
        with open(yaml_file, 'r') as f:
            self.objects_data = yaml.safe_load(f)['objects']

    def publish_pointcloud(self):
        cloud_data = []
        total_points = 0

        # Define translation once (used for both point cloud and markers)
        translation = np.array([-0.0, +0.0, 0])

        for ply_file in self.ply_files:
            full_path = os.path.join(self.folder_path, ply_file)
            pcd = o3d.io.read_point_cloud(full_path)
            pts = np.asarray(pcd.points)

            # Apply translation
            pts = pts + translation

            # Handle colors
            if len(pcd.colors) == 0:
                col = np.array([random.random(), random.random(), random.random()])
                colors = (np.tile(col, (pts.shape[0], 1)) * 255).astype(np.uint8)
            else:
                colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

            total_points += pts.shape[0]
            self.get_logger().info(f'{ply_file}: {pts.shape[0]} pts')

            # Pack RGB
            rgb_uint32 = np.array([
                struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
                for r, g, b in colors
            ], dtype=np.uint32)

            combined = np.column_stack((pts, rgb_uint32))
            cloud_data.extend(combined.tolist())

        # Publish PointCloud2
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
        ]
        pc2_msg = pc2.create_cloud(header, fields, cloud_data)
        self.pc_pub.publish(pc2_msg)
        self.get_logger().info(f'Published total of {total_points:,} points')

        # Publish markers
        marker_array = MarkerArray()
        marker_id = 0

        # Markers from YAML - APPLY SAME TRANSLATION AS POINT CLOUD
        for obj_id, obj in self.objects_data.items():
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = header.stamp
            marker.ns = "object_centers"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            # Apply translation to match point cloud
            marker.pose.position.x = obj['position']['x'] + translation[0]
            marker.pose.position.y = obj['position']['y'] + translation[1]
            marker.pose.position.z = obj['position']['z'] + translation[2]
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)

        # Green markers at (0.3, 0.5, 0) and (-0.3, 0.5, 0)
        green_positions = [ (0.2, 0.4, 0), (-0.2, 0.4   , 0) ]
        for pos in green_positions:
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = header.stamp
            marker.ns = "green_markers"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)

        # Blue marker at camera pose
        cam_pos = (-0.000, 0.811, 0.711)
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = header.stamp
        marker.ns = "camera_marker"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(cam_pos[0])
        marker.pose.position.y = float(cam_pos[1])
        marker.pose.position.z = float(cam_pos[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.07
        marker.scale.y = 0.07
        marker.scale.z = 0.07
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    folder = "/home/liam/git/Capstone_Converting_Natural_Langauge_To_Robot_Control/full_integration/out/scene_objects"
    yaml_file = "/home/liam/git/Capstone_Converting_Natural_Langauge_To_Robot_Control/full_integration/out/scene.yamle"

    node = MultiObjectPublisher(folder, yaml_file)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()