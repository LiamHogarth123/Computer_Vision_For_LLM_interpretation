#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation Script for SAM+CLIP+Point Cloud System

This script evaluates your vision system's performance by comparing
YAML outputs against manually measured ground truth.

Usage:
  1. Run your vision system to generate objects.yaml
  2. Create ground_truth.yaml with manual measurements
  3. Run this script: python evaluate_system.py --results objects.yaml --ground-truth ground_truth.yaml

Author: Evaluation Module
Date: 2024
"""

import argparse
import os
import json
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Any

class VisionSystemEvaluator:
    """
    Comprehensive evaluator for SAM+CLIP+Point Cloud vision system
    """
    
    def __init__(self, results_yaml_path: str, ground_truth_yaml_path: str, output_dir: str = "evaluation_output"):
        self.results_path = results_yaml_path
        self.gt_path = ground_truth_yaml_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.results = self.load_yaml(results_yaml_path)
        self.ground_truth = self.load_yaml(ground_truth_yaml_path)
        
        # Evaluation metrics storage
        self.metrics = {
            'detection': {},
            'classification': {},
            'localization': {},
            'dimensions': {},
            'overall': {}
        }
        
        print(f"\n{'='*70}")
        print("VISION SYSTEM EVALUATION")
        print(f"{'='*70}")
        print(f"Results file: {results_yaml_path}")
        print(f"Ground truth: {ground_truth_yaml_path}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*70}\n")
    
    def load_yaml(self, filepath: str) -> Dict:
        """Load YAML file"""
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    
    def save_json(self, data: Dict, filename: str):
        """Save data as JSON"""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[SAVED] {filepath}")
    
    # ==================== DETECTION METRICS ====================
    
    def evaluate_detection(self) -> Dict:
        """
        Evaluate object detection performance
        Returns: precision, recall, F1-score, mAP
        """
        print("\n" + "="*70)
        print("EVALUATING DETECTION PERFORMANCE")
        print("="*70)
        
        pred_objects = self.results.get('objects', {})
        gt_objects = self.ground_truth.get('objects', {})
        
        n_predicted = len(pred_objects)
        n_ground_truth = len(gt_objects)
        
        # Match predictions to ground truth
        matches, unmatched_pred, unmatched_gt = self.match_objects(
            pred_objects, gt_objects, distance_threshold=0.10  # 10cm threshold
        )
        
        true_positives = len(matches)
        false_positives = len(unmatched_pred)
        false_negatives = len(unmatched_gt)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate mAP (using IoU in 3D space)
        average_precision = self.calculate_average_precision(matches, pred_objects, gt_objects)
        
        # Store matches as list of dicts for safer access
        match_list = []
        for pred_id, gt_id, distance in matches:
            match_list.append({
                'pred_id': pred_id,
                'gt_id': gt_id,
                'distance': distance
            })
        
        detection_metrics = {
            'n_predicted': n_predicted,
            'n_ground_truth': n_ground_truth,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mAP': average_precision,
            'matches': match_list,  # Changed to list of dicts
            'unmatched_predictions': unmatched_pred,
            'unmatched_ground_truth': unmatched_gt
        }
        
        self.metrics['detection'] = detection_metrics
        
        # Print results
        print(f"Predicted objects: {n_predicted}")
        print(f"Ground truth objects: {n_ground_truth}")
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")
        print(f"\nPrecision: {precision:.3f} ({precision*100:.1f}%)")
        print(f"Recall: {recall:.3f} ({recall*100:.1f}%)")
        print(f"F1-Score: {f1_score:.3f}")
        print(f"mAP@0.10m: {average_precision:.3f}")
        
        return detection_metrics
    
    def match_objects(self, pred_objects: Dict, gt_objects: Dict, 
                     distance_threshold: float = 0.10) -> Tuple[List, List, List]:
        """
        Match predicted objects to ground truth using Hungarian algorithm
        
        Returns:
            matches: List of (pred_id, gt_id, distance) tuples
            unmatched_pred: List of unmatched prediction IDs
            unmatched_gt: List of unmatched ground truth IDs
        """
        pred_ids = list(pred_objects.keys())
        gt_ids = list(gt_objects.keys())
        
        print(f"\n[DEBUG] Matching {len(pred_ids)} predictions to {len(gt_ids)} ground truth objects")
        print(f"[DEBUG] Prediction IDs: {pred_ids[:5]}...")  # Show first 5
        print(f"[DEBUG] Ground truth IDs: {gt_ids[:5]}...")
        
        if len(pred_ids) == 0 or len(gt_ids) == 0:
            return [], pred_ids, gt_ids
        
        # Build distance matrix
        n_pred = len(pred_ids)
        n_gt = len(gt_ids)
        distance_matrix = np.zeros((n_pred, n_gt))
        
        for i, pred_id in enumerate(pred_ids):
            try:
                pred_pos = np.array([
                    pred_objects[pred_id]['position']['x'],
                    pred_objects[pred_id]['position']['y'],
                    pred_objects[pred_id]['position']['z']
                ])
            except (KeyError, TypeError) as e:
                print(f"[WARNING] Could not extract position for prediction '{pred_id}': {e}")
                pred_pos = np.array([999, 999, 999])  # Far away position
            
            for j, gt_id in enumerate(gt_ids):
                try:
                    gt_pos = np.array([
                        gt_objects[gt_id]['position']['x'],
                        gt_objects[gt_id]['position']['y'],
                        gt_objects[gt_id]['position']['z']
                    ])
                except (KeyError, TypeError) as e:
                    print(f"[WARNING] Could not extract position for ground truth '{gt_id}': {e}")
                    gt_pos = np.array([999, 999, 999])
                
                distance = np.linalg.norm(pred_pos - gt_pos)
                distance_matrix[i, j] = distance
        
        # Hungarian algorithm for optimal matching
        row_indices, col_indices = linear_sum_assignment(distance_matrix)
        
        matches = []
        matched_pred = set()
        matched_gt = set()
        
        for i, j in zip(row_indices, col_indices):
            distance = distance_matrix[i, j]
            if distance <= distance_threshold:
                matches.append((pred_ids[i], gt_ids[j], distance))
                matched_pred.add(pred_ids[i])
                matched_gt.add(gt_ids[j])
        
        unmatched_pred = [pid for pid in pred_ids if pid not in matched_pred]
        unmatched_gt = [gid for gid in gt_ids if gid not in matched_gt]
        
        return matches, unmatched_pred, unmatched_gt
    
    def calculate_average_precision(self, matches: List, pred_objects: Dict, gt_objects: Dict) -> float:
        """Calculate mean Average Precision"""
        if len(matches) == 0:
            return 0.0
        
        # Simple AP calculation: percentage of correct matches
        return len(matches) / len(gt_objects) if len(gt_objects) > 0 else 0.0
    
    # ==================== CLASSIFICATION METRICS ====================
    
    def evaluate_classification(self) -> Dict:
        """
        Evaluate CLIP classification accuracy
        """
        print("\n" + "="*70)
        print("EVALUATING CLASSIFICATION PERFORMANCE")
        print("="*70)
        
        matches = self.metrics['detection']['matches']
        
        correct = 0
        incorrect = 0
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        confidence_scores = []
        classification_details = []
        
        pred_objects = self.results.get('objects', {})
        gt_objects = self.ground_truth.get('objects', {})
        
        if not matches:
            print("[WARNING] No matched objects for classification evaluation")
            return {
                'total_classifications': 0,
                'correct': 0,
                'incorrect': 0,
                'accuracy': 0.0,
                'mean_confidence': 0.0,
                'confusion_matrix': {},
                'details': []
            }
        
        for match in matches:
            pred_id = match['pred_id']
            gt_id = match['gt_id']
            distance = match['distance']
            
            # Check if IDs exist in dictionaries
            if pred_id not in pred_objects:
                print(f"[WARNING] Predicted object ID '{pred_id}' not found in results")
                continue
            if gt_id not in gt_objects:
                print(f"[WARNING] Ground truth object ID '{gt_id}' not found")
                continue
            
            pred_label = self.clean_label(pred_objects[pred_id].get('name', 'unknown'))
            gt_label = self.clean_label(gt_objects[gt_id].get('name', 'unknown'))
            confidence = pred_objects[pred_id].get('confidence', 0.0)
            
            confidence_scores.append(confidence)
            confusion_matrix[gt_label][pred_label] += 1
            
            is_correct = self.labels_match(pred_label, gt_label)
            
            if is_correct:
                correct += 1
            else:
                incorrect += 1
            
            classification_details.append({
                'predicted': pred_label,
                'ground_truth': gt_label,
                'correct': is_correct,
                'confidence': confidence,
                'distance_m': distance
            })
        
        accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
        mean_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        classification_metrics = {
            'total_classifications': correct + incorrect,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': accuracy,
            'mean_confidence': mean_confidence,
            'confusion_matrix': dict(confusion_matrix),
            'details': classification_details
        }
        
        self.metrics['classification'] = classification_metrics
        
        # Print results
        print(f"Total classifications: {correct + incorrect}")
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Mean confidence: {mean_confidence:.3f}")
        
        # Print confusion matrix
        if confusion_matrix:
            print("\nConfusion Matrix:")
            header = "GT \\ Pred"
            print(f"{header:>15}", end='')
            pred_labels = sorted(set(label for gt_dict in confusion_matrix.values() for label in gt_dict.keys()))
            for pred_label in pred_labels:
                print(f"{pred_label:>12}", end='')
            print()
            
            for gt_label in sorted(confusion_matrix.keys()):
                print(f"{gt_label:>15}", end='')
                for pred_label in pred_labels:
                    count = confusion_matrix[gt_label].get(pred_label, 0)
                    print(f"{count:>12}", end='')
                print()
        
        return classification_metrics
    
    def clean_label(self, label: str) -> str:
        """Clean and normalize label"""
        return label.lower().strip().replace('_', ' ')
    
    def labels_match(self, pred_label: str, gt_label: str) -> bool:
        """Check if labels match (fuzzy matching)"""
        pred = self.clean_label(pred_label)
        gt = self.clean_label(gt_label)
        
        # Exact match
        if pred == gt:
            return True
        
        # Substring match
        if pred in gt or gt in pred:
            return True
        
        # Color variants (e.g., "red dice" matches "dice")
        pred_words = set(pred.split())
        gt_words = set(gt.split())
        common_words = pred_words & gt_words
        
        # If they share significant words, consider it a match
        if len(common_words) > 0 and (len(common_words) / min(len(pred_words), len(gt_words))) > 0.5:
            return True
        
        return False
    
    # ==================== LOCALIZATION METRICS ====================
    
    def evaluate_localization(self) -> Dict:
        """
        Evaluate 3D position accuracy
        """
        print("\n" + "="*70)
        print("EVALUATING LOCALIZATION ACCURACY")
        print("="*70)
        
        matches = self.metrics['detection']['matches']
        pred_objects = self.results['objects']
        gt_objects = self.ground_truth['objects']
        
        position_errors = []
        position_errors_xyz = {'x': [], 'y': [], 'z': []}
        
        for match in matches:
            pred_id = match['pred_id']
            gt_id = match['gt_id']
            
            try:
                pred_pos = np.array([
                    pred_objects[pred_id]['position']['x'],
                    pred_objects[pred_id]['position']['y'],
                    pred_objects[pred_id]['position']['z']
                ])
                gt_pos = np.array([
                    gt_objects[gt_id]['position']['x'],
                    gt_objects[gt_id]['position']['y'],
                    gt_objects[gt_id]['position']['z']
                ])
            except KeyError as e:
                print(f"[WARNING] Missing position data for pred={pred_id} or gt={gt_id}: {e}")
                continue
            
            error = np.linalg.norm(pred_pos - gt_pos)
            position_errors.append(error * 1000)  # Convert to mm
            
            # Per-axis errors
            error_xyz = pred_pos - gt_pos
            position_errors_xyz['x'].append(abs(error_xyz[0]) * 1000)
            position_errors_xyz['y'].append(abs(error_xyz[1]) * 1000)
            position_errors_xyz['z'].append(abs(error_xyz[2]) * 1000)
        
        if position_errors:
            localization_metrics = {
                'mean_error_mm': np.mean(position_errors),
                'std_error_mm': np.std(position_errors),
                'median_error_mm': np.median(position_errors),
                'min_error_mm': np.min(position_errors),
                'max_error_mm': np.max(position_errors),
                'percentile_95_mm': np.percentile(position_errors, 95),
                'per_axis': {
                    'x_mean_mm': np.mean(position_errors_xyz['x']),
                    'y_mean_mm': np.mean(position_errors_xyz['y']),
                    'z_mean_mm': np.mean(position_errors_xyz['z']),
                    'x_std_mm': np.std(position_errors_xyz['x']),
                    'y_std_mm': np.std(position_errors_xyz['y']),
                    'z_std_mm': np.std(position_errors_xyz['z'])
                },
                'all_errors_mm': position_errors
            }
        else:
            localization_metrics = {
                'mean_error_mm': 0,
                'std_error_mm': 0,
                'note': 'No matched objects for localization evaluation'
            }
        
        self.metrics['localization'] = localization_metrics
        
        # Print results
        if position_errors:
            print(f"Position Error (mm):")
            print(f"  Mean:   {localization_metrics['mean_error_mm']:.2f} ± {localization_metrics['std_error_mm']:.2f}")
            print(f"  Median: {localization_metrics['median_error_mm']:.2f}")
            print(f"  Range:  [{localization_metrics['min_error_mm']:.2f}, {localization_metrics['max_error_mm']:.2f}]")
            print(f"  95th percentile: {localization_metrics['percentile_95_mm']:.2f}")
            print(f"\nPer-axis errors (mm):")
            print(f"  X: {localization_metrics['per_axis']['x_mean_mm']:.2f} ± {localization_metrics['per_axis']['x_std_mm']:.2f}")
            print(f"  Y: {localization_metrics['per_axis']['y_mean_mm']:.2f} ± {localization_metrics['per_axis']['y_std_mm']:.2f}")
            print(f"  Z: {localization_metrics['per_axis']['z_mean_mm']:.2f} ± {localization_metrics['per_axis']['z_std_mm']:.2f}")
        
        return localization_metrics
    
    # ==================== DIMENSION METRICS ====================
    
    def evaluate_dimensions(self) -> Dict:
        """
        Evaluate dimension estimation accuracy
        """
        print("\n" + "="*70)
        print("EVALUATING DIMENSION ESTIMATION")
        print("="*70)
        
        matches = self.metrics['detection']['matches']
        pred_objects = self.results['objects']
        gt_objects = self.ground_truth['objects']
        
        dimension_errors = {'length': [], 'width': [], 'height': []}
        
        for match in matches:
            pred_id = match['pred_id']
            gt_id = match['gt_id']
            
            try:
                pred_dims = pred_objects[pred_id]['dimensions']
                gt_dims = gt_objects[gt_id]['dimensions']
            except KeyError as e:
                print(f"[WARNING] Missing dimension data for pred={pred_id} or gt={gt_id}: {e}")
                continue
            
            for dim in ['length', 'width', 'height']:
                error = abs(pred_dims[dim] - gt_dims[dim]) * 1000  # mm
                dimension_errors[dim].append(error)
        
        dimension_metrics = {}
        for dim in ['length', 'width', 'height']:
            if dimension_errors[dim]:
                dimension_metrics[f'{dim}_mean_error_mm'] = np.mean(dimension_errors[dim])
                dimension_metrics[f'{dim}_std_error_mm'] = np.std(dimension_errors[dim])
                dimension_metrics[f'{dim}_median_error_mm'] = np.median(dimension_errors[dim])
        
        self.metrics['dimensions'] = dimension_metrics
        
        # Print results
        print("Dimension Errors (mm):")
        for dim in ['length', 'width', 'height']:
            if dimension_errors[dim]:
                mean_key = f'{dim}_mean_error_mm'
                std_key = f'{dim}_std_error_mm'
                print(f"  {dim.capitalize():6}: {dimension_metrics[mean_key]:.2f} ± {dimension_metrics[std_key]:.2f}")
        
        return dimension_metrics
    
    # ==================== VISUALIZATION ====================
    
    def plot_position_error_distribution(self):
        """Plot position error histogram"""
        if 'all_errors_mm' not in self.metrics['localization']:
            return
        
        errors = self.metrics['localization']['all_errors_mm']
        
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Position Error (mm)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Position Error Distribution', fontsize=14, fontweight='bold')
        plt.axvline(np.mean(errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f} mm')
        plt.axvline(np.median(errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f} mm')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = self.output_dir / 'position_error_distribution.png'
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[SAVED] {filepath}")
    
    def plot_confusion_matrix(self):
        """Plot classification confusion matrix"""
        if 'confusion_matrix' not in self.metrics['classification']:
            return
        
        cm = self.metrics['classification']['confusion_matrix']
        if not cm:
            return
        
        # Get all labels
        all_labels = sorted(set(list(cm.keys()) + [label for gt_dict in cm.values() for label in gt_dict.keys()]))
        
        # Build matrix
        matrix = np.zeros((len(all_labels), len(all_labels)))
        for i, gt_label in enumerate(all_labels):
            for j, pred_label in enumerate(all_labels):
                matrix[i, j] = cm.get(gt_label, {}).get(pred_label, 0)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, interpolation='nearest', cmap='Blues')
        plt.title('Classification Confusion Matrix', fontsize=14, fontweight='bold')
        plt.colorbar()
        
        tick_marks = np.arange(len(all_labels))
        plt.xticks(tick_marks, all_labels, rotation=45, ha='right')
        plt.yticks(tick_marks, all_labels)
        
        # Add text annotations
        thresh = matrix.max() / 2
        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                plt.text(j, i, int(matrix[i, j]),
                        ha="center", va="center",
                        color="white" if matrix[i, j] > thresh else "black")
        
        plt.ylabel('Ground Truth', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        
        filepath = self.output_dir / 'confusion_matrix.png'
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[SAVED] {filepath}")
    
    def plot_per_axis_errors(self):
        """Plot per-axis position errors"""
        if 'per_axis' not in self.metrics['localization']:
            return
        
        per_axis = self.metrics['localization']['per_axis']
        
        axes = ['x', 'y', 'z']
        means = [per_axis[f'{ax}_mean_mm'] for ax in axes]
        stds = [per_axis[f'{ax}_std_mm'] for ax in axes]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        x_pos = np.arange(len(axes))
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Axis', fontsize=12)
        ax.set_ylabel('Mean Error (mm)', fontsize=12)
        ax.set_title('Position Error by Axis', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([ax.upper() for ax in axes])
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        filepath = self.output_dir / 'per_axis_errors.png'
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[SAVED] {filepath}")
    
    # ==================== REPORT GENERATION ====================
    
    def generate_latex_table(self):
        """Generate LaTeX table for report"""
        latex = []
        
        # Detection table
        latex.append("% Detection Performance Table")
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Object Detection Performance}")
        latex.append("\\begin{tabular}{lc}")
        latex.append("\\hline")
        latex.append("Metric & Value \\\\")
        latex.append("\\hline")
        
        det = self.metrics['detection']
        latex.append(f"Precision & {det['precision']:.3f} ({det['precision']*100:.1f}\\%) \\\\")
        latex.append(f"Recall & {det['recall']:.3f} ({det['recall']*100:.1f}\\%) \\\\")
        latex.append(f"F1-Score & {det['f1_score']:.3f} \\\\")
        latex.append(f"mAP@0.10m & {det['mAP']:.3f} \\\\")
        
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\label{tab:detection}")
        latex.append("\\end{table}")
        latex.append("")
        
        # Localization table
        latex.append("% Localization Accuracy Table")
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{3D Localization Accuracy}")
        latex.append("\\begin{tabular}{lc}")
        latex.append("\\hline")
        latex.append("Metric & Value (mm) \\\\")
        latex.append("\\hline")
        
        loc = self.metrics['localization']
        if 'mean_error_mm' in loc:
            latex.append(f"Mean Error & {loc['mean_error_mm']:.2f} $\\pm$ {loc['std_error_mm']:.2f} \\\\")
            latex.append(f"Median Error & {loc['median_error_mm']:.2f} \\\\")
            latex.append(f"95th Percentile & {loc['percentile_95_mm']:.2f} \\\\")
        
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\label{tab:localization}")
        latex.append("\\end{table}")
        
        # Save to file
        filepath = self.output_dir / 'latex_tables.tex'
        with open(filepath, 'w') as f:
            f.write('\n'.join(latex))
        print(f"[SAVED] {filepath}")
    
    def generate_summary_report(self):
        """Generate comprehensive text report"""
        lines = []
        
        lines.append("="*70)
        lines.append("VISION SYSTEM EVALUATION REPORT")
        lines.append("="*70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Results file: {self.results_path}")
        lines.append(f"Ground truth: {self.gt_path}")
        lines.append("="*70)
        lines.append("")
        
        # Detection
        lines.append("1. DETECTION PERFORMANCE")
        lines.append("-" * 70)
        det = self.metrics['detection']
        lines.append(f"   Objects predicted: {det['n_predicted']}")
        lines.append(f"   Ground truth objects: {det['n_ground_truth']}")
        lines.append(f"   True Positives: {det['true_positives']}")
        lines.append(f"   False Positives: {det['false_positives']}")
        lines.append(f"   False Negatives: {det['false_negatives']}")
        lines.append(f"   Precision: {det['precision']:.3f} ({det['precision']*100:.1f}%)")
        lines.append(f"   Recall: {det['recall']:.3f} ({det['recall']*100:.1f}%)")
        lines.append(f"   F1-Score: {det['f1_score']:.3f}")
        lines.append(f"   mAP@0.10m: {det['mAP']:.3f}")
        lines.append("")
        
        # Classification
        lines.append("2. CLASSIFICATION ACCURACY")
        lines.append("-" * 70)
        cls = self.metrics['classification']
        lines.append(f"   Total classifications: {cls['total_classifications']}")
        lines.append(f"   Correct: {cls['correct']}")
        lines.append(f"   Incorrect: {cls['incorrect']}")
        lines.append(f"   Accuracy: {cls['accuracy']:.3f} ({cls['accuracy']*100:.1f}%)")
        lines.append(f"   Mean confidence: {cls['mean_confidence']:.3f}")
        lines.append("")
        
        # Localization
        lines.append("3. LOCALIZATION ACCURACY")
        lines.append("-" * 70)
        loc = self.metrics['localization']
        if 'mean_error_mm' in loc:
            lines.append(f"   Position Error:")
            lines.append(f"     Mean: {loc['mean_error_mm']:.2f} ± {loc['std_error_mm']:.2f} mm")
            lines.append(f"     Median: {loc['median_error_mm']:.2f} mm")
            lines.append(f"     Range: [{loc['min_error_mm']:.2f}, {loc['max_error_mm']:.2f}] mm")
            lines.append(f"     95th percentile: {loc['percentile_95_mm']:.2f} mm")
            lines.append(f"   Per-axis errors:")
            lines.append(f"     X: {loc['per_axis']['x_mean_mm']:.2f} ± {loc['per_axis']['x_std_mm']:.2f} mm")
            lines.append(f"     Y: {loc['per_axis']['y_mean_mm']:.2f} ± {loc['per_axis']['y_std_mm']:.2f} mm")
            lines.append(f"     Z: {loc['per_axis']['z_mean_mm']:.2f} ± {loc['per_axis']['z_std_mm']:.2f} mm")
        lines.append("")
        
        # Dimensions
        if self.metrics['dimensions']:
            lines.append("4. DIMENSION ESTIMATION")
            lines.append("-" * 70)
            dims = self.metrics['dimensions']
            for dim in ['length', 'width', 'height']:
                mean_key = f'{dim}_mean_error_mm'
                std_key = f'{dim}_std_error_mm'
                if mean_key in dims:
                    lines.append(f"   {dim.capitalize()}: {dims[mean_key]:.2f} ± {dims[std_key]:.2f} mm")
            lines.append("")
        
        lines.append("="*70)
        
        report_text = '\n'.join(lines)
        
        # Print to console
        print("\n" + report_text)
        
        # Save to file
        filepath = self.output_dir / 'evaluation_report.txt'
        with open(filepath, 'w') as f:
            f.write(report_text)
        print(f"\n[SAVED] {filepath}")
        
        return report_text
    
    # ==================== MAIN EVALUATION ====================
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("\n" + "="*70)
        print("STARTING FULL EVALUATION")
        print("="*70)
        
        # Run all evaluations
        self.evaluate_detection()
        self.evaluate_classification()
        self.evaluate_localization()
        self.evaluate_dimensions()
        
        # Generate visualizations
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        self.plot_position_error_distribution()
        self.plot_confusion_matrix()
        self.plot_per_axis_errors()
        
        # Generate reports
        print("\n" + "="*70)
        print("GENERATING REPORTS")
        print("="*70)
        self.generate_latex_table()
        self.save_json(self.metrics, 'evaluation_metrics.json')
        self.generate_summary_report()
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE")
        print("="*70)
        print(f"All outputs saved to: {self.output_dir}")
        print("="*70)


def create_ground_truth_template():
    """Create a template ground truth YAML file"""
    template = {
        'timestamp': datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        'note': 'Manually measure object positions and dimensions with calipers/ruler',
        'objects': {
            'blue_dice_001': {
                'name': 'blue dice',
                'position': {
                    'x': 0.150,  # meters - measure from robot base
                    'y': 0.250,
                    'z': 0.020
                },
                'dimensions': {
                    'length': 0.016,  # meters - measure with calipers
                    'width': 0.016,
                    'height': 0.016
                },
                'orientation': {
                    'yaw': 0.0  # degrees
                }
            },
            'red_dice_002': {
                'name': 'red dice',
                'position': {
                    'x': 0.100,
                    'y': 0.300,
                    'z': 0.020
                },
                'dimensions': {
                    'length': 0.016,
                    'width': 0.016,
                    'height': 0.016
                },
                'orientation': {
                    'yaw': 45.0
                }
            }
        }
    }
    
    with open('ground_truth_template.yaml', 'w') as f:
        yaml.dump(template, f, indent=2, sort_keys=False)
    
    print("Created ground_truth_template.yaml")
    print("Fill in actual measurements for your objects!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vision System Performance")
    parser.add_argument('--results', type=str,
                       help='Path to results YAML file (from your vision system)')
    parser.add_argument('--ground-truth', type=str,
                       help='Path to ground truth YAML file (manually measured)')
    parser.add_argument('--output', type=str, default='evaluation_output',
                       help='Output directory for evaluation results')
    parser.add_argument('--create-template', action='store_true',
                       help='Create a ground truth template file and exit')
    
    args = parser.parse_args()
    
    if args.create_template:
        create_ground_truth_template()
        return
    
    # Now require results and ground-truth if not creating template
    if not args.results or not args.ground_truth:
        parser.error("--results and --ground-truth are required (unless using --create-template)")
    
    # Run evaluation
    evaluator = VisionSystemEvaluator(
        results_yaml_path=args.results,
        ground_truth_yaml_path=args.ground_truth,
        output_dir=args.output
    )
    
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()