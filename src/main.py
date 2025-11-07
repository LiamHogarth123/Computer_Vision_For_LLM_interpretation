#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Computer Vision System: AR Tag Calibration + SAM/CLIP Object Detection
Phase 1: AR tag calibration for camera pose estimation (or load pre-calibrated transform)
Phase 2: SAM segmentation + CLIP labeling + 3D point cloud mapping

Usage:
  # Use saved calibration:
  python point_cloud_masking.py --mode live --sam-checkpoint sam_vit_b.pth --save-prefix out/scene --use-saved-transform camera_calibration_transform.npy
  
  # Run new calibration:
  python point_cloud_masking.py --mode live --sam-checkpoint sam_vit_b.pth --save-prefix out/scene
  
  # From bag file with saved calibration:
  python point_cloud_masking.py --mode bag --bag scene.bag --sam-checkpoint sam_vit_b.pth --use-saved-transform camera_calibration_transform.npy
"""

import argparse
import os
import time
from collections import defaultdict
from datetime import datetime
from math import atan2

import numpy as np
import cv2
import cv2.aruco as aruco
from PIL import Image
import yaml
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation

import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import clip
import open3d as o3d
import tf_transformations
import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # Replace with your message type

# ==================== CONFIGURATION ====================
class Config:
    """Central configuration for entire vision system"""
    
    # AR Tag Calibration
    AR_MARKER_LENGTH = 0.10  # meters
    AR_TAGS_POSITIONS = {
        1: np.array([-0.25, 0.46, 0]),  # Left tag
        2: np.array([0.25, 0.46, 0]),   # Right tag
    }
    CALIBRATION_DURATION = 3.0  # seconds
    MIN_CALIBRATION_DETECTIONS = 10
    
    # Camera intrinsics (adjust to your camera)
    CAMERA_MATRIX = np.array([
        [615, 0, 320],
        [0, 615, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    DIST_COEFFS = np.zeros(5)
    
    # SAM parameters
    SAM_POINTS_PER_SIDE = 12
    SAM_PRED_IOU_THRESH = 0.92
    SAM_STABILITY_THRESH = 0.92
    SAM_BOX_NMS_THRESH = 0.5
    SAM_MIN_MASK_AREA = 2500
    
    # CLIP candidate labels
    CLIP_LABELS = [
        # "a photo of a lemon, a yellow citrus fruit",
        # "a photo of a yellow fruit",
        # "a photo of a citrus fruit",
        # "a photo of an apple",
        # "a photo of a red apple",
        # "a photo of a tape measure, a measuring tool",
        # "a photo of a computer keyboard",
        # "a photo of a computer mouse",
        # "a photo of a blue and black computer mouse",
        # "a photo of a calculator",
        # "a photo of a cardboard box with the realsense logo",
        "a photo of a wooden desk",
        # "a photo of a coffee cup",
        # "a photo of a smartphone",
        # "a photo of a silver kitchen knife",
        # "a photo of a fork",
        # "a photo of a silver fork",
        # "a photo of a remote controller",
        # "a photo of a green and black coffee cup",
        # "a photo of a black coffee cup",
        # "a photo of blue case",
        # "a photo of a dark blue case",
        # "a photo of a kincrome case",
        # "a photo of red and black eraser",
        # "a photo of a redbull can",
        # "a photo of a blue and silver redbull can",
        # "a photo of a red and blue and silver redbull can",
        # "a photo of a can",
        "a photo of a unknown object",
        "a photo of a blue dice",
        "a photo of a yellow dice",
        "a photo of a red dice",
        # "a photo of a black dot",
        "a photo of a Ar tag",
        # "a photo of a carboard tabletop",
        # "a photo of a aluminum tabletop",
        # "a photo of a brown tabletop",
        # "a photo of carboard",
        # "a photo of a yellow jar of vegmite",
        "a photo of a wooden table",
        "a photo of  wooden table top surface",
        # "a photo of a hammer",
        # "a photo of a egg",
        # "a photo of a yellow and black tape measure",
        # "a photo of a white candle",
        # "a photo of a lit white candle",
        # "a photo of a green and blue spirit can",
    
        # "a photo of a yellow tape measure",
        # "a photo of a checkerfboard pattern"
    ]
    
    # Point cloud filtering
    DEPTH_TOLERANCE = 0.015  # meters
    PLANE_DISTANCE_THRESH = 0.008
    VOXEL_DOWNSAMPLE_SIZE = 0.003

# ==================== AR TAG CALIBRATION ====================
class ARTagCalibrator:
    """Handles AR tag detection and camera pose estimation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
    def estimate_pose(self, corners):
        """Estimate pose of detected markers"""
        if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
            return cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.config.AR_MARKER_LENGTH,
                self.config.CAMERA_MATRIX,
                self.config.DIST_COEFFS
            )
        else:
            objp = np.array([
                [-self.config.AR_MARKER_LENGTH/2,  self.config.AR_MARKER_LENGTH/2, 0],
                [ self.config.AR_MARKER_LENGTH/2,  self.config.AR_MARKER_LENGTH/2, 0],
                [ self.config.AR_MARKER_LENGTH/2, -self.config.AR_MARKER_LENGTH/2, 0],
                [-self.config.AR_MARKER_LENGTH/2, -self.config.AR_MARKER_LENGTH/2, 0],
            ], dtype=np.float32)
            rvecs, tvecs = [], []
            for c in corners:
                _, rvec, tvec = cv2.solvePnP(
                    objp, c[0],
                    self.config.CAMERA_MATRIX,
                    self.config.DIST_COEFFS
                )
                rvecs.append(rvec)
                tvecs.append(tvec)
            return np.array(rvecs), np.array(tvecs), None
    
    def rvec_tvec_to_matrix(self, rvec, tvec):
        """Convert rotation vector and translation to 4x4 transform"""
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T
    
    def compute_camera_pose(self, rvecs, tvecs, tag_ids):
        """Compute camera pose from detected AR tags"""
        camera_positions = []
        camera_orientations = []
        
        for i, tag_id in enumerate(tag_ids):
            if tag_id not in self.config.AR_TAGS_POSITIONS:
                continue
            
            # Transform from camera to tag (from solvePnP)
            T_camera_tag = self.rvec_tvec_to_matrix(rvecs[i], tvecs[i])
            
            # Transform from world to tag
            T_world_tag = np.eye(4)
            T_world_tag[:3, 3] = self.config.AR_TAGS_POSITIONS[tag_id]
            
            # Camera pose in world frame
            # T_camera_tag gives us the tag position in camera frame
            # We want the camera position in world frame
            T_tag_camera = np.linalg.inv(T_camera_tag)
            T_world_camera = T_world_tag @ T_tag_camera
            
            camera_positions.append(T_world_camera[:3, 3])
            camera_orientations.append(T_world_camera[:3, :3])
        
        if not camera_positions:
            return None
        
        # Average position
        avg_position = np.mean(camera_positions, axis=0)
        
        # Use first tag's orientation
        R = camera_orientations[0]
        
        # Build final transform
        T_camera = np.eye(4)
        T_camera[:3, :3] = R
        T_camera[:3, 3] = avg_position
        
        return T_camera
    
    def load_saved_transform(self, filepath: str) -> np.ndarray:
        """
        Load a pre-calibrated camera transform from file.
        Returns 4x4 transformation matrix.
        """
        print("\n" + "="*70)
        print("PHASE 1: LOADING SAVED CALIBRATION")
        print("="*70)
        print(f"Loading transform from: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibration file not found: {filepath}")
        
        T_camera = np.load(filepath)
        
        if T_camera.shape != (4, 4):
            raise ValueError(f"Invalid transform shape: {T_camera.shape}, expected (4, 4)")
        
        # Print camera pose
        camera_pos = T_camera[:3, 3]
        camera_quat = tf_transformations.quaternion_from_matrix(T_camera)
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(camera_quat)
        
        print(f"✓ Loaded camera pose:")
        print(f"  Position: x={camera_pos[0]:.3f}, y={camera_pos[1]:.3f}, z={camera_pos[2]:.3f}")
        print(f"  Orientation: roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°")
        print("="*70)
        
        return T_camera
    
    def run_calibration(self, pipeline, align, args) -> np.ndarray:
        """
        Run calibration routine to estimate camera pose.
        Returns 4x4 transformation matrix or None if failed.
        """
        print("\n" + "="*70)
        print("PHASE 1: AR TAG CALIBRATION")
        print("="*70)
        print(f"Duration: {self.config.CALIBRATION_DURATION}s")
        print(f"Looking for tags: {list(self.config.AR_TAGS_POSITIONS.keys())}")
        print("Position camera to see AR tags clearly...")
        print()
        
        start_time = time.time()
        pose_estimates = []
        
        while time.time() - start_time < self.config.CALIBRATION_DURATION:
            frames = pipeline.wait_for_frames(args.timeout_ms)
            if not frames:
                continue
            
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            # Detect AR tags
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)
            
            if ids is not None and len(ids) > 0:
                ids = ids.flatten()
                rvecs, tvecs, _ = self.estimate_pose(corners)
                T_camera = self.compute_camera_pose(rvecs, tvecs, ids)
                
                if T_camera is not None:
                    pose_estimates.append(T_camera)
                    
                    # Visual feedback
                    for i in range(len(ids)):
                        cv2.drawFrameAxes(
                            color_image,
                            self.config.CAMERA_MATRIX,
                            self.config.DIST_COEFFS,
                            rvecs[i], tvecs[i], 0.05
                        )
                    
                    status = f"Calibrating... {len(pose_estimates)} samples | Tags: {ids.tolist()}"
                    cv2.putText(color_image, status, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(color_image, "No AR tags detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow("AR Tag Calibration", color_image)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
                break
        
        cv2.destroyAllWindows()
        
        # Average pose estimates
        if len(pose_estimates) >= self.config.MIN_CALIBRATION_DETECTIONS:
            print(f"\n✓ Calibration successful: {len(pose_estimates)} samples")
            
            # Average positions
            avg_position = np.mean([T[:3, 3] for T in pose_estimates], axis=0)
            
            # Use last transform's rotation (or could average quaternions)
            T_final = pose_estimates[-1].copy()
            T_final[:3, 3] = avg_position
            
            # Print result
            camera_quat = tf_transformations.quaternion_from_matrix(T_final)
            roll, pitch, yaw = tf_transformations.euler_from_quaternion(camera_quat)
            print(f"Camera pose: x={avg_position[0]:.3f}, y={avg_position[1]:.3f}, z={avg_position[2]:.3f}")
            print(f"Orientation: roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°")
            print("="*70)
            
            return T_final
        else:
            print(f"\n❌ Calibration failed: only {len(pose_estimates)} samples "
                  f"(need {self.config.MIN_CALIBRATION_DETECTIONS})")
            return None

# ==================== SAM + CLIP DETECTION ====================
def load_sam_clip_models(sam_checkpoint, sam_type="vit_b", device="cuda"):
    """Load SAM and CLIP models"""
    print("\n" + "="*70)
    print("PHASE 2: LOADING AI MODELS")
    print("="*70)
    
    # SAM
    print(f"[INFO] Loading SAM model: {sam_type}")
    sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    # CLIP
    print("[INFO] Loading CLIP model: ViT-B/32")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    print("="*70)
    return sam, clip_model, preprocess

def segment_image_sam(sam_model, image, config):
    """Run SAM segmentation"""
    print("\n" + "="*70)
    print("PHASE 3: SAM SEGMENTATION")
    print("="*70)
    
    sam_start = time.time()

    mask_generator = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=config.SAM_POINTS_PER_SIDE,
        pred_iou_thresh=config.SAM_PRED_IOU_THRESH,
        stability_score_thresh=config.SAM_STABILITY_THRESH,
        box_nms_thresh=config.SAM_BOX_NMS_THRESH,
        min_mask_region_area=config.SAM_MIN_MASK_AREA,
    )
    
    print("[INFO] Running SAM segmentation...")
    masks = mask_generator.generate(image)
    sam_time = time.time() - sam_start
    print(f"⏱️  SAM: {sam_time:.3f}s")

    print(f"[INFO] Found {len(masks)} masks")
    print("="*70)
    
    return masks

def label_masks_with_clip(masks, image, clip_model, preprocess, labels, device="cuda"):
    """Label each mask using CLIP"""
    print("\n" + "="*70)
    print("PHASE 4: CLIP LABELING")
    print("="*70)
    
    text_features = clip.tokenize(labels).to(device)
    with torch.no_grad():
        text_embeddings = clip_model.encode_text(text_features)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    
    labeled_masks = []
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    for i, mask_info in enumerate(masks):
        mask = mask_info['segmentation']
        bbox = mask_info['bbox']
        x, y, w, h = bbox
        
        # Extract ROI
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(image.shape[1], int(x + w)), min(image.shape[0], int(y + h))
        roi = pil_image.crop((x1, y1, x2, y2))
        
        # CLIP inference
        roi_tensor = preprocess(roi).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = clip_model.encode_image(roi_tensor)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        
        # Compute similarities
        similarities = (image_embedding @ text_embeddings.T).squeeze(0)
        best_idx = similarities.argmax().item()
        confidence = similarities[best_idx].item()
        
        labeled_masks.append({
            'mask': mask,
            'bbox': bbox,
            'label': labels[best_idx],
            'confidence': confidence
        })
        
        if (i + 1) % 10 == 0 or (i + 1) == len(masks):
            print(f"[INFO] Labeled {i+1}/{len(masks)} masks")
    
    print("="*70)
    return labeled_masks

def overlay_masks(image, labeled_masks):
    """Create visualization with colored masks and labels"""
    out = image.copy()
    
    for idx, m in enumerate(labeled_masks):
        mask = np.array(m['mask'], dtype=bool)
        
        # Generate bright random color
        color = tuple(map(int, np.random.randint(80, 255, 3).tolist()))
        
        # Apply semi-transparent colored mask
        out[mask] = (0.5 * out[mask] + 0.5 * np.array(color)).astype(np.uint8)
        
        # Draw contours
        mask_uint8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
        
        # Add label at top-left of bounding box
        if 'label' in m and 'bbox' in m:
            label_text = m['label'].split(',')[0].replace('a photo of', '').strip()
            x, y, _, _ = m['bbox']
            cv2.putText(out, label_text, (int(x), max(int(y)-5, 0)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return out

def save_individual_object_images(image, labeled_masks, output_dir):
    """Save cropped images of each detected object for easy identification"""
    obj_images_dir = os.path.join(output_dir, "object_images")
    os.makedirs(obj_images_dir, exist_ok=True)
    
    saved_images = []
    
    for idx, obj in enumerate(labeled_masks):
        mask = obj['mask']
        bbox = obj['bbox']
        x, y, w, h = [int(v) for v in bbox]
        
        # Add padding
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        
        # Crop image
        cropped = image[y1:y2, x1:x2].copy()
        
        # Create masked version (object isolated on white background)
        mask_crop = mask[y1:y2, x1:x2]
        masked_img = np.ones_like(cropped) * 255
        masked_img[mask_crop] = cropped[mask_crop]
        
        # Save both versions
        label_text = obj['label'].split(',')[0].replace('a photo of', '').strip().replace(' ', '_')
        
        # Full crop
        crop_path = os.path.join(obj_images_dir, f"{idx:03d}_{label_text}_crop.png")
        cv2.imwrite(crop_path, cropped)
        
        # Masked version
        masked_path = os.path.join(obj_images_dir, f"{idx:03d}_{label_text}_masked.png")
        cv2.imwrite(masked_path, masked_img)
        
        saved_images.append({
            'id': idx,
            'label': label_text,
            'crop_path': crop_path,
            'masked_path': masked_path
        })
    
    print(f"[INFO] Saved {len(saved_images)} object images to {obj_images_dir}")
    return saved_images

# ==================== POINT CLOUD PROCESSING ====================
def project_depth_to_camera(depth_frame, intrinsics):
    """Project depth frame to 3D points in camera frame"""
    depth_array = np.asanyarray(depth_frame.get_data()) * depth_frame.get_units()
    h, w = depth_array.shape
    
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy
    
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_array
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = vertices[:, 2] > 0
    
    uv = np.stack([u / w, v / h], axis=-1).reshape(-1, 2)
    
    return vertices[valid], uv[valid]

def build_lookup_from_uv(verts, uv, img_w, img_h):
    """Build UV to 3D point lookup"""
    lookup = {}
    uv_px = (uv * np.array([img_w, img_h])).astype(int)
    uv_px[:, 0] = np.clip(uv_px[:, 0], 0, img_w - 1)
    uv_px[:, 1] = np.clip(uv_px[:, 1], 0, img_h - 1)
    
    for i, (u, v) in enumerate(uv_px):
        key = (v, u)
        if key not in lookup:
            lookup[key] = []
        lookup[key].append(i)
    
    return lookup

def collect_mask_points(mask, lookup, vertices):
    """Extract 3D points for a mask"""
    ys, xs = np.where(mask)
    point_indices = []
    
    for y, x in zip(ys, xs):
        key = (int(y), int(x))
        if key in lookup:
            point_indices.extend(lookup[key])
    
    if len(point_indices) == 0:
        return np.empty((0, 3))
    
    return vertices[point_indices]

def detect_table_plane(points, distance_thresh=0.01, ransac_n=3, num_iterations=1000):
    """Detect horizontal table plane using RANSAC"""
    if points.shape[0] < 100:
        return None, []
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_thresh,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    if plane_model is not None:
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)
        
        # Check if it's horizontal
        if abs(normal[2]) > 0.9:
            return plane_model, inliers
    
    return None, []

def save_colored_points(points, color, filepath):
    """Save point cloud with uniform color"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.tile(color, (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filepath, pcd)

def compute_tabletop_dimensions_and_yaw(points):
    """Compute object dimensions and orientation for tabletop objects"""
    if points.shape[0] < 10:
        return {
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0
        }, {
            "length": 0.0,
            "width": 0.0,
            "height": 0.0
        }
    
    # Project to XY plane
    xy_points = points[:, :2]
    
    # Compute PCA
    centroid_xy = xy_points.mean(axis=0)
    centered = xy_points - centroid_xy
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Primary direction
    primary_vec = eigenvectors[:, 0]
    yaw = atan2(primary_vec[1], primary_vec[0])
    
    # Project to principal axes
    projected = centered @ eigenvectors
    
    # Compute dimensions
    length = np.ptp(projected[:, 0])
    width = np.ptp(projected[:, 1])
    height = np.ptp(points[:, 2])
    
    return {
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": float(np.degrees(yaw))
    }, {
        "length": float(length),
        "width": float(width),
        "height": float(height)
    }


def filter_objects_near_robot(scene_objects, robot_position=(0, 0, 0), exclusion_radius=0.25):
    """
    Filter out objects within exclusion_radius of the robot arm position.
    
    Args:
        scene_objects: List of object dictionaries with 'centroid' key
        robot_position: Tuple (x, y, z) of robot arm base position
        exclusion_radius: Radius in meters to exclude objects (default 0.25m)
    
    Returns:
        filtered_objects: List of objects outside the exclusion zone
        excluded_count: Number of objects filtered out
    """
    robot_pos = np.array(robot_position)
    filtered_objects = []
    excluded_count = 0
    
    for obj in scene_objects:
        obj_centroid = np.array(obj['centroid'])
        distance = np.linalg.norm(obj_centroid - robot_pos)
        
        if distance >= exclusion_radius:
            filtered_objects.append(obj)
        else:
            excluded_count += 1
            print(f"  [FILTERED] {obj['label'].split(',')[0].strip()}: "
                  f"too close to robot arm ({distance:.3f}m < {exclusion_radius}m)")
    
    return filtered_objects, excluded_count


def filter_objects_outside_workspace(scene_objects, verts_world, plane_model=None, 
                                    margin=0.05, z_tolerance=0.10):
    """
    Filter out objects that are outside the detected rectangular workspace/tabletop.
    
    Args:
        scene_objects: List of object dictionaries with 'centroid' key
        verts_world: All point cloud vertices in world frame
        plane_model: Pre-detected plane model [a, b, c, d] or None to detect
        margin: XY margin in meters to expand workspace bounds (default 0.05m)
        z_tolerance: Z tolerance in meters above/below plane (default 0.10m)
    
    Returns:
        filtered_objects: List of objects within the workspace
        excluded_count: Number of objects filtered out
        workspace_bounds: Dict with 'x_min', 'x_max', 'y_min', 'y_max', 'z_plane'
    """
    # Detect or use existing plane model
    if plane_model is None:
        plane_model, inliers = detect_table_plane(verts_world)
        if plane_model is None:
            print("[WARN] Could not detect table plane, skipping workspace filtering")
            return scene_objects, 0, None
    else:
        # Re-detect inliers for existing plane
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(verts_world)
        _, inliers = pcd_temp.segment_plane(
            distance_threshold=Config.PLANE_DISTANCE_THRESH,
            ransac_n=3,
            num_iterations=200
        )
    
    # Get plane inlier points
    plane_points = verts_world[inliers]
    
    # Calculate workspace bounds from plane points (rectangular bounding box in XY)
    x_min, x_max = plane_points[:, 0].min() - margin, plane_points[:, 0].max() + margin
    y_min, y_max = plane_points[:, 1].min() - margin, plane_points[:, 1].max() + margin
    
    # Calculate plane Z height (average of inlier Z values)
    z_plane = plane_points[:, 2].mean()
    
    workspace_bounds = {
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
        'z_plane': z_plane,
        'z_min': z_plane - z_tolerance,
        'z_max': z_plane + z_tolerance
    }
    
    print(f"[INFO] Workspace bounds: X=[{x_min:.2f}, {x_max:.2f}], "
          f"Y=[{y_min:.2f}, {y_max:.2f}], Z_plane={z_plane:.2f}")
    
    # Filter objects
    filtered_objects = []
    excluded_count = 0
    
    for obj in scene_objects:
        cx, cy, cz = obj['centroid']
        
        # Check if object is within XY bounds
        in_x_bounds = x_min <= cx <= x_max
        in_y_bounds = y_min <= cy <= y_max
        
        # Check if object is near the plane Z height
        in_z_bounds = (z_plane - z_tolerance) <= cz <= (z_plane + z_tolerance)
        
        if in_x_bounds and in_y_bounds and in_z_bounds:
            filtered_objects.append(obj)
        else:
            excluded_count += 1
            reason = []
            if not in_x_bounds: reason.append("X out of bounds")
            if not in_y_bounds: reason.append("Y out of bounds")
            if not in_z_bounds: reason.append("Z not on table plane")
            
            print(f"  [FILTERED] {obj['label'].split(',')[0].strip()}: "
                  f"outside workspace ({', '.join(reason)})")
    
    return filtered_objects, excluded_count, workspace_bounds

def wait_for_ready_signal(node):
    """Wait until a ROS2 message is received before continuing"""
    node.get_logger().info("Waiting for ready signal...")
    ready_msg = None
    
    def callback(msg):
        nonlocal ready_msg
        ready_msg = msg
    
    subscription = node.create_subscription(
        String,
        '/ready_topic',
        callback,
        10
    )
    
    while ready_msg is None and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        break
    
    node.destroy_subscription(subscription)
    if ready_msg is not None:
        node.get_logger().info(f"Received ready signal: {ready_msg.data}")
    
    return ready_msg


# ==================== MAIN PIPELINE ====================
def main():
    parser = argparse.ArgumentParser(description="Integrated AR Tag + SAM/CLIP Vision System")
    parser.add_argument("--mode", choices=["live", "bag"], default="live",
                       help="Input mode: live camera or bag file")
    parser.add_argument("--bag", type=str, help="Path to ROS bag file (required if mode=bag)")
    parser.add_argument("--sam-checkpoint", type=str, required=True,
                       help="Path to SAM checkpoint (.pth file)")
    parser.add_argument("--sam-type", type=str, default="vit_b",
                       choices=["vit_b", "vit_l", "vit_h"],
                       help="SAM model type")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for inference (cuda/cpu)")
    parser.add_argument("--save-prefix", type=str, default="scene",
                       help="Prefix for output files")
    parser.add_argument("--timeout-ms", type=int, default=5000,
                       help="Frame timeout in milliseconds")
    parser.add_argument("--use-saved-transform", type=str, default=None,
                       help="Path to saved calibration transform (.npy file). If provided, skips AR tag calibration.")
    
    args = parser.parse_args()
    

    # Validate arguments
    if args.mode == "bag" and not args.bag:
        parser.error("--bag is required when --mode=bag")
    
    # Initialize config and models
    config = Config()
    calibrator = ARTagCalibrator(config)

    rclpy.init()
    node = Node('vision_processor')
    
    # Setup output directory
    output_dir = os.path.dirname(args.save_prefix) or "."
    os.makedirs(output_dir, exist_ok=True)
    per_obj_dir = os.path.join(output_dir, "objects")
    os.makedirs(per_obj_dir, exist_ok=True)
    
    # Setup RealSense
    pipeline = rs.pipeline()
    rs_config = rs.config()
    
    if args.mode == "bag":
        rs_config.enable_device_from_file(args.bag, repeat_playback=False)
    else:
        rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    profile = pipeline.start(rs_config)
    align = rs.align(rs.stream.color)
    
    try:
        # ==================== PHASE 1: CALIBRATION ====================
        if args.use_saved_transform:
            # Load saved calibration
            T_world_camera = calibrator.load_saved_transform(args.use_saved_transform)
        else:
            # Run new calibration
            T_world_camera = calibrator.run_calibration(pipeline, align, args)
        
        if T_world_camera is None:
            print("\n❌ ERROR: Camera calibration failed. Cannot proceed.")
            return
        
        # ==================== PHASE 2-4: AI MODELS + DETECTION ====================
        sam_model, clip_model, preprocess = load_sam_clip_models(
            args.sam_checkpoint, args.sam_type, args.device
        )
        
        
        
        while True:

                # Capture frame for detection
            print("\n[INFO] Capturing frame for object detection...")
            frames = pipeline.wait_for_frames(args.timeout_ms)
            frames = align.process(frames)
            
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                raise RuntimeError("Failed to capture frames")
            
            color_bgr = np.asanyarray(color_frame.get_data())
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            
            # SAM segmentation
            masks = segment_image_sam(sam_model, color_rgb, config)
            
            # CLIP labeling
            clip_start = time.time()
            labels = label_masks_with_clip(
                masks, color_bgr, clip_model, preprocess,
                config.CLIP_LABELS, args.device
            )
            clip_time = time.time() - clip_start
            # timing_results["clip_seconds"] = clip_time
            # timing_results["objects_labeled"] = len(labels)
            print(f"⏱️  CLIP: {clip_time:.3f}s")

            # ==================== PHASE 5: 3D POINT CLOUD ====================
            print("\n" + "="*70)
            print("PHASE 4: BUILDING 3D POINT CLOUD")
            print("="*70)
            pc_start = time.time()
            
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            verts_cam, uv = project_depth_to_camera(depth_frame, intrinsics)
            
            # Transform to world frame using homogeneous coordinates
            # Convert to homogeneous coordinates (add column of ones)
            verts_h = np.hstack([verts_cam, np.ones((verts_cam.shape[0], 1))])
            
            # Apply transformation matrix
            # T_world_camera represents the camera pose, we use it directly
            verts_world_h = (T_world_camera @ verts_h.T).T
            
            # Convert back to 3D coordinates
            verts_world = verts_world_h[:, :3]

            camera_pos = T_world_camera[:3, 3].copy()

            # x_offset = 0.01  # Adjust this value as needed (positive moves toward robot)
            # # x_offset = camera_pos[0]
            # verts_world[:, 0] += x_offset




            # y_offset = -0.078  # Adjust this value as needed (positive moves toward robot)
            # y_offset = -camera_pos[1]
            # print(y_offset)

            # verts_world[:, 1] += y_offset

            z_offset = 0.04   # for example, shift upward by 10 cm
            # z_offset = camera_pos[2]
            # print(z_offset)
            verts_world[:, 2] += z_offset

            
            
            print(f"[INFO] Point cloud: {verts_world.shape[0]} points")
            
            # Save full cloud
            img_h, img_w = color_bgr.shape[:2]
            colors = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB) / 255.0
            
            full_pcd = o3d.geometry.PointCloud()
            full_pcd.points = o3d.utility.Vector3dVector(verts_world)
            
            uv_px = (uv * np.array([img_w, img_h])).astype(int)
            uv_px[:, 0] = np.clip(uv_px[:, 0], 0, img_w - 1)
            uv_px[:, 1] = np.clip(uv_px[:, 1], 0, img_h - 1)
            full_colors = colors[uv_px[:, 1], uv_px[:, 0], :]
            full_pcd.colors = o3d.utility.Vector3dVector(full_colors)
            
            full_cloud_path = args.save_prefix + "_full_cloud.ply"
            o3d.io.write_point_cloud(full_cloud_path, full_pcd)
            print(f"[INFO] Saved full point cloud: {full_cloud_path}")
            
            # Detect table plane
            plane_model, inliers = detect_table_plane(verts_world)
            if plane_model is not None and len(inliers) > 0:
                plane_normal = np.array(plane_model[:3])
                plane_normal /= np.linalg.norm(plane_normal)
                if abs(plane_normal[2]) > 0.9:
                    print(f"[INFO] Table plane detected ({len(inliers)} inlier points)")
            
            # Build UV lookup
            lookup = build_lookup_from_uv(verts_world, uv, img_w, img_h)
            
            # ==================== PHASE 5: EXTRACT OBJECTS ====================
            print()
            print("="*70)
            print("PHASE 5: EXTRACTING OBJECT POINT CLOUDS")
            print("="*70)
            
            scene_objects = []
            for i, obj in enumerate(labels):
                mask_arr = np.array(obj['mask'], dtype=np.uint8).astype(bool)
                
                # Erode mask to remove boundary noise
                kernel = np.ones((5, 5), np.uint8)
                mask_eroded = cv2.erode(mask_arr.astype(np.uint8), kernel, iterations=1).astype(bool)
                
                # Collect 3D points
                obj_points = collect_mask_points(mask_eroded, lookup, verts_world)
                if obj_points.shape[0] == 0:
                    continue
                
                # Depth filtering
                z_median = np.median(obj_points[:, 2])
                depth_mask = np.abs(obj_points[:, 2] - z_median) < config.DEPTH_TOLERANCE
                obj_points = obj_points[depth_mask]
                
                if obj_points.shape[0] == 0:
                    continue
                
                # Remove table plane if detected
                if plane_model is not None and obj_points.shape[0] >= 3:  # Add point count check
                    pcd_temp = o3d.geometry.PointCloud()
                    pcd_temp.points = o3d.utility.Vector3dVector(obj_points)
                    plane_model_obj, inliers_obj = pcd_temp.segment_plane(
                        distance_threshold=config.PLANE_DISTANCE_THRESH,
                        ransac_n=3,
                        num_iterations=200
                    )
                    if plane_model_obj is not None:
                        a, b, c, d = plane_model_obj
                        normal = np.array([a, b, c])
                        if abs(normal[2]) > 0.9:
                            pcd_temp = pcd_temp.select_by_index(inliers_obj, invert=True)
                            obj_points = np.asarray(pcd_temp.points)
                
                # Downsample if too many points
                if obj_points.shape[0] > 20000:
                    pcd_down = o3d.geometry.PointCloud()
                    pcd_down.points = o3d.utility.Vector3dVector(obj_points)
                    pcd_down = pcd_down.voxel_down_sample(voxel_size=config.VOXEL_DOWNSAMPLE_SIZE)
                    obj_points = np.asarray(pcd_down.points)
                
                if obj_points.size == 0:
                    print(f"[WARN] Object {i} '{obj.get('label', 'unknown')}' has no points, skipping")
                    continue
                
                # Compute properties
                color = np.random.rand(3)
                centroid = obj_points.mean(axis=0).tolist()
                min_xyz = obj_points.min(axis=0).tolist()
                max_xyz = obj_points.max(axis=0).tolist()
                bbox_3d = {"min": min_xyz, "max": max_xyz}
                
                # Compute orientation and dimensions
                orientation, dimensions = compute_tabletop_dimensions_and_yaw(obj_points)
                
                # Save individual object PLY
                label_name = obj['label'].split(',')[0].replace('a photo of', '').strip().replace(' ', '_')
                # ply_name = os.path.join(per_obj_dir, f"object_{i:03d}_{label_name}.ply")
                # save_colored_points(obj_points, color, ply_name)
                
                save_folder = "/home/liam/git/Capstone_Converting_Natural_Langauge_To_Robot_Control/full_integration/out/scene_objects"
                ply_name = os.path.join(save_folder, f"object_{i:03d}_{label_name}.ply")

                scene_objects.append({
                    "id": i,
                    "label": obj.get("label", "unknown"),
                    "confidence": obj.get("confidence", 0.0),
                    "bbox_2d": obj.get("bbox"),
                    "num_points": int(obj_points.shape[0]),
                    "centroid": centroid,
                    "bbox_3d": bbox_3d,
                    "ply_file": ply_name,
                    "color": color.tolist(),
                    "orientation": orientation,
                    "dimensions": dimensions
                })
                
                print(f"  [{i+1}/{len(labels)}] {label_name}: "
                    f"{obj_points.shape[0]} pts, "
                    f"pos=[{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")
            
            print(f"\n[INFO] Extracted {len(scene_objects)} objects")
            
            pc_time = time.time() - pc_start
            # timing_results["pointcloud_seconds"] = pc_time
            print(f"⏱️  Point cloud: {pc_time:.3f}s")
            # ==================== PHASE 6: SAVE RESULTS ====================
            print()
            print("="*70)
            print("PHASE 6: SAVING RESULTS")
            print("="*70)
            
            # scene_objects, robot_excluded = filter_objects_near_robot(
            #     scene_objects, 
            #     robot_position=(0, 0, 0), 
            #     exclusion_radius=0.25
            # )
            # print(f"[INFO] Filtered {robot_excluded} objects near robot arm")
            
            # Filter 2: Remove objects outside workspace
            scene_objects, workspace_excluded, workspace_bounds = filter_objects_outside_workspace(
                scene_objects,
                verts_world,
                plane_model=plane_model,
                margin=0.05,
                z_tolerance=0.10
            )
            print(f"[INFO] Filtered {workspace_excluded} objects outside workspace")
            print(f"[INFO] {len(scene_objects)} objects remaining after filtering")


            # Extract the IDs of remaining objects
            remaining_object_ids = {obj['id'] for obj in scene_objects}

            # Filter the labels list to match filtered scene_objects
            filtered_labels = [labels[i] for i in remaining_object_ids if i < len(labels)]


            # Save overlay image (filtered objects only)
            overlay = overlay_masks(color_bgr, filtered_labels)
            overlay_path = args.save_prefix + "_overlay.png"
            cv2.imwrite(overlay_path, overlay)
            print(f"[INFO] Saved overlay: {overlay_path} ({len(filtered_labels)} objects)")

            # Optionally save unfiltered overlay for comparison
            overlay_all = overlay_masks(color_bgr, labels)
            overlay_all_path = args.save_prefix + "_overlay_all_objects.png"
            cv2.imwrite(overlay_all_path, overlay_all)
            print(f"[INFO] Saved unfiltered overlay: {overlay_all_path} ({len(labels)} objects)")
                    
            # Save individual object images for easy identification
            object_images = save_individual_object_images(color_bgr, labels, output_dir)
            print(f"[INFO] Individual object images saved in {output_dir}/object_images/")
            
            # Build structured YAML
            objects_dict = {}
            z_threshold = 0.04  # Threshold for small z values (configurable)
            z_offset = 0.04     # Offset to apply to small z values

            for obj in scene_objects:
                label_clean = obj['label'].split(',')[0].replace('a photo of', '').strip()
                obj_id = f"{label_clean.replace(' ', '_')}_{obj['id']:03d}"
                cx, cy, cz = obj["centroid"]
                
                # Check if z value is small
                is_graspable = cz >= z_threshold
                
                # Apply offset if z is small
                if not is_graspable:
                    cz = z_offset
                
                objects_dict[obj_id] = {
                    "name": label_clean,
                    "description": obj["label"],
                    "category": "unknown",
                    "position": {
                        "x": float(cx),
                        "y": float(cy),
                        "z": float(cz)
                    },
                    "orientation": {
                        "roll": obj["orientation"]["roll"],
                        "pitch": obj["orientation"]["pitch"],
                        "yaw": obj["orientation"]["yaw"]
                    },
                    "dimensions": {
                        "length": obj["dimensions"]["length"],
                        "width": obj["dimensions"]["width"],
                        "height": obj["dimensions"]["height"]
                    },
                    "properties": {
                        "graspable": is_graspable,
                        "moveable": True,
                        "weight": "unknown",
                        "color": "unknown",
                        "material": "unknown"
                    },
                    "confidence": float(obj["confidence"]),
                    "num_points": obj["num_points"],
                    "ply_file": obj["ply_file"],
                    "last_seen": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                }
            # Add camera pose to YAML
            # Apply same Y offset that was applied to point cloud
            camera_pos = T_world_camera[:3, 3].copy()
            camera_pos[1] += 1  # Apply same y_offset as line 708
            camera_quat = tf_transformations.quaternion_from_matrix(T_world_camera)
            
            scene_yaml = {
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "camera_pose": {
                    "position": {
                        "x": float(camera_pos[0]),
                        "y": float(camera_pos[1]),
                        "z": float(camera_pos[2])
                    },
                    "orientation": {
                        "x": float(camera_quat[0]),
                        "y": float(camera_quat[1]),
                        "z": float(camera_quat[2]),
                        "w": float(camera_quat[3])
                    }
                },
                "objects": objects_dict
            }
            
            # yaml_path = args.save_prefix + ".yaml"
            yaml_path = "/home/liam/git/HRC_Capstone/config" "objects.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(scene_yaml, f, indent=2, sort_keys=False)
            print(f"[INFO] Saved scene YAML: {yaml_path}")
            
            # Summary
            print()
            print("="*70)
            print("✅ PIPELINE COMPLETE")
            print("="*70)
            print(f"Objects detected: {len(scene_objects)}")
            print(f"Camera pose: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}]")
            print(f"Output prefix: {args.save_prefix}")
            print("="*70)

            wait_for_ready_signal(node)  # Add 'node' here! ()
        

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if pipeline:
            print("\n[INFO] Stopping pipeline...")
            pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()