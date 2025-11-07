#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Computer Vision System: AR Tag Calibration + SAM/CLIP Object Detection
WITH PERFORMANCE TIMING MEASUREMENTS

This version includes comprehensive timing to measure:
- SAM generation time vs batch size (points_per_side)
- Total pipeline time
- Individual phase times
- Accuracy metrics

Usage:
  python main_with_timing.py --mode live --sam-checkpoint sam_vit_b.pth --save-prefix out/scene
"""

import argparse
import os
import time
from collections import defaultdict
from datetime import datetime
from math import atan2
import json

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
from std_msgs.msg import String

# ==================== TIMING UTILITIES ====================
class PerformanceTimer:
    """Track and record performance metrics for different batch sizes"""
    
    def __init__(self):
        self.timings = {}
        self.current_phase = None
        self.phase_start = None
        self.total_start = None
        
    def start_total(self):
        """Start timing the entire pipeline"""
        self.total_start = time.time()
        
    def start_phase(self, phase_name):
        """Start timing a specific phase"""
        self.current_phase = phase_name
        self.phase_start = time.time()
        
    def end_phase(self):
        """End timing current phase"""
        if self.current_phase and self.phase_start:
            elapsed = time.time() - self.phase_start
            self.timings[self.current_phase] = elapsed
            print(f"⏱️  [{self.current_phase}] completed in {elapsed:.3f}s")
            self.current_phase = None
            self.phase_start = None
            return elapsed
        return 0
    
    def get_total_time(self):
        """Get total elapsed time"""
        if self.total_start:
            return time.time() - self.total_start
        return 0
    
    def save_results(self, output_path, accuracy_metrics=None, batch_config=None):
        """Save timing results to JSON file"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_time": self.get_total_time(),
            "phase_timings": self.timings,
            "batch_config": batch_config or {},
            "accuracy_metrics": accuracy_metrics or {}
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Performance results saved to: {output_path}")
        return results
    
    def print_summary(self):
        """Print timing summary to terminal"""
        print("\n" + "="*70)
        print("⏱️  PERFORMANCE TIMING SUMMARY")
        print("="*70)
        for phase, duration in self.timings.items():
            print(f"  {phase:.<50} {duration:>8.3f}s")
        print(f"  {'TOTAL TIME':.<50} {self.get_total_time():>8.3f}s")
        print("="*70)

# ==================== ACCURACY TRACKING ====================
class AccuracyTracker:
    """Track accuracy metrics for the detection system"""
    
    def __init__(self):
        self.metrics = {
            "total_objects_detected": 0,
            "objects_after_filtering": 0,
            "total_masks_generated": 0,
            "false_positives": 0,  # Extra masks added
            "missed_objects": 0,
            "table_detected": False
        }
    
    def update(self, **kwargs):
        """Update metrics"""
        self.metrics.update(kwargs)
    
    def calculate_success_rate(self):
        """Calculate overall success metrics"""
        total = self.metrics["total_objects_detected"]
        filtered = self.metrics["objects_after_filtering"]
        false_pos = self.metrics["false_positives"]
        
        return {
            "detection_accuracy": filtered / total if total > 0 else 0,
            "false_positive_rate": false_pos / total if total > 0 else 0,
            "objects_detected": filtered,
            "extra_masks": false_pos
        }
    
    def get_metrics(self):
        """Get all metrics"""
        return {**self.metrics, **self.calculate_success_rate()}

# ==================== CONFIGURATION ====================
class Config:
    """Central configuration for entire vision system"""
    
    # AR Tag Calibration
    AR_MARKER_LENGTH = 0.10
    AR_TAGS_POSITIONS = {
        1: np.array([-0.25, 0.46, 0]),
        2: np.array([0.25, 0.46, 0]),
    }
    CALIBRATION_DURATION = 3.0
    MIN_CALIBRATION_DETECTIONS = 10
    
    # Camera intrinsics
    CAMERA_MATRIX = np.array([
        [615, 0, 320],
        [0, 615, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    DIST_COEFFS = np.zeros(5)
    
    # SAM parameters - THESE ARE WHAT YOU'LL VARY FOR TESTING
    SAM_POINTS_PER_SIDE = 32  # Try: 16, 32, 48, 64
    SAM_PRED_IOU_THRESH = 0.92
    SAM_STABILITY_THRESH = 0.92
    SAM_BOX_NMS_THRESH = 0.5
    SAM_MIN_MASK_AREA = 2500
    
    # CLIP candidate labels
    CLIP_LABELS = [
        "a photo of a redbull can",
        "a photo of a blue and silver redbull can",
        "a photo of a red and blue and silver redbull can",
        "a photo of a can",
        "a photo of a unknown object",
        "a photo of a Ar tag",
        "a photo of a wooden table",
        "a photo of wooden table top surface",
        "a photo of a yellow and black tape measure",
        "a photo of a white candle",
        "a photo of a lit white candle",
        "a photo of a green and blue spirit can",
    ]
    
    # Point cloud filtering
    DEPTH_TOLERANCE = 0.015
    PLANE_DISTANCE_THRESH = 0.008
    VOXEL_DOWNSAMPLE_SIZE = 0.003

# ... [Keep all the rest of your original code classes: ARTagCalibrator, etc.] ...
# For brevity, I'm showing just the key modifications to main()

def main():
    """Main pipeline with timing measurements"""
    
    # ==================== INITIALIZE TIMING ====================
    timer = PerformanceTimer()
    accuracy = AccuracyTracker()
    timer.start_total()
    
    parser = argparse.ArgumentParser(description="Integrated AR + SAM + CLIP Vision System with Timing")
    parser.add_argument("--mode", choices=["live", "bag"], default="live")
    parser.add_argument("--bag", help="Path to ROS bag file (if mode=bag)")
    parser.add_argument("--sam-checkpoint", required=True, help="Path to SAM checkpoint")
    parser.add_argument("--sam-type", default="vit_b", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--save-prefix", default="out/scene", help="Output prefix")
    parser.add_argument("--use-saved-transform", help="Path to saved camera transform (.npy)")
    
    # TIMING-SPECIFIC ARGUMENTS
    parser.add_argument("--batch-size", type=int, help="Override SAM points_per_side for testing")
    parser.add_argument("--timing-output", default="timing_results.json", help="Output file for timing results")
    
    args = parser.parse_args()
    
    config = Config()
    
    # Override batch size if specified
    if args.batch_size:
        config.SAM_POINTS_PER_SIDE = args.batch_size
        print(f"🔧 Using custom batch size (points_per_side): {args.batch_size}")
    
    batch_config = {
        "points_per_side": config.SAM_POINTS_PER_SIDE,
        "pred_iou_thresh": config.SAM_PRED_IOU_THRESH,
        "stability_thresh": config.SAM_STABILITY_THRESH,
        "min_mask_area": config.SAM_MIN_MASK_AREA
    }
    
    # ... [Your existing setup code] ...
    
    try:
        # ==================== PHASE 1: AR TAG CALIBRATION ====================
        timer.start_phase("Phase 1: AR Tag Calibration")
        # ... [Your existing AR calibration code] ...
        timer.end_phase()
        
        # ==================== PHASE 2: CAPTURE FRAME ====================
        timer.start_phase("Phase 2: Frame Capture")
        # ... [Your existing frame capture code] ...
        timer.end_phase()
        
        # ==================== PHASE 3: SAM SEGMENTATION ====================
        timer.start_phase("Phase 3: SAM Segmentation")
        print(f"\n🔍 Running SAM with points_per_side={config.SAM_POINTS_PER_SIDE}")
        
        # TIME THE CRITICAL SAM GENERATION STEP
        sam_start = time.time()
        
        sam = sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint)
        sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=config.SAM_POINTS_PER_SIDE,
            pred_iou_thresh=config.SAM_PRED_IOU_THRESH,
            stability_score_thresh=config.SAM_STABILITY_THRESH,
            box_nms_thresh=config.SAM_BOX_NMS_THRESH,
            min_mask_region_area=config.SAM_MIN_MASK_AREA
        )
        
        masks = mask_generator.generate(color_rgb)
        
        sam_time = time.time() - sam_start
        timer.timings["SAM_generation_only"] = sam_time
        
        print(f"✅ SAM generated {len(masks)} masks in {sam_time:.3f}s")
        print(f"   Throughput: {len(masks)/sam_time:.2f} masks/second")
        
        accuracy.update(total_masks_generated=len(masks))
        timer.end_phase()
        
        # ==================== PHASE 4: CLIP LABELING ====================
        timer.start_phase("Phase 4: CLIP Labeling")
        # ... [Your existing CLIP code] ...
        
        accuracy.update(total_objects_detected=len(labels))
        timer.end_phase()
        
        # ==================== PHASE 5: 3D POINT CLOUD ====================
        timer.start_phase("Phase 5: 3D Point Cloud Generation")
        # ... [Your existing point cloud code] ...
        timer.end_phase()
        
        # ==================== PHASE 6: FILTERING & SAVE ====================
        timer.start_phase("Phase 6: Object Filtering & Save")
        # ... [Your existing filtering code] ...
        
        # Track filtering results
        objects_before = len(scene_objects)
        # ... [after your filtering] ...
        objects_after = len(scene_objects)
        
        accuracy.update(
            objects_after_filtering=objects_after,
            false_positives=max(0, objects_after - 5)  # Assuming 5 expected objects
        )
        
        timer.end_phase()
        
        # ==================== FINAL TIMING REPORT ====================
        timer.print_summary()
        
        # Print detailed SAM performance
        print("\n" + "="*70)
        print("🎯 SAM PERFORMANCE ANALYSIS")
        print("="*70)
        print(f"Batch Size (points_per_side): {config.SAM_POINTS_PER_SIDE}")
        print(f"SAM Generation Time: {sam_time:.3f}s")
        print(f"Masks Generated: {len(masks)}")
        print(f"Throughput: {len(masks)/sam_time:.2f} masks/second")
        print(f"Average Time per Mask: {sam_time/len(masks)*1000:.1f}ms")
        print("="*70)
        
        # Print accuracy metrics
        print("\n" + "="*70)
        print("📊 ACCURACY METRICS")
        print("="*70)
        metrics = accuracy.get_metrics()
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:.<50} {value:>8.2%}" if value <= 1 else f"{key:.<50} {value:>8.2f}")
            else:
                print(f"{key:.<50} {value:>8}")
        print("="*70)
        
        # Save results to JSON
        results_path = args.timing_output
        timer.save_results(
            results_path,
            accuracy_metrics=metrics,
            batch_config=batch_config
        )
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if pipeline:
            pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()