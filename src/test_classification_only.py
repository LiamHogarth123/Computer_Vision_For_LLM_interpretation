#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification-Only Evaluation Test

This script ONLY tests CLIP's labeling accuracy, ignoring position/dimensions.
Perfect for when you know exactly what objects are in the scene and just want
to verify that CLIP labels them correctly.

Usage:
    # Create test config with expected objects
    python test_classification_only.py --create-config
    
    # Run classification test
    python test_classification_only.py \
        --results /path/to/objects.yaml \
        --expected-objects test_config.yaml \
        --output classification_test_results
"""

import argparse
import yaml
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt


class ClassificationTester:
    """
    Tests ONLY classification/labeling accuracy.
    Ignores positions, dimensions, and all 3D information.
    """
    
    def __init__(self, results_yaml: str, expected_yaml: str, output_dir: str = "classification_test"):
        self.results_path = results_yaml
        self.expected_path = expected_yaml
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        with open(results_yaml, 'r') as f:
            self.results = yaml.safe_load(f)
        
        with open(expected_yaml, 'r') as f:
            self.expected = yaml.safe_load(f)
        
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'total_expected': 0,
            'total_detected': 0,
            'correct_labels': 0,
            'incorrect_labels': 0,
            'missing_objects': 0,
            'extra_detections': 0,
            'details': []
        }
        
        print(f"\n{'='*70}")
        print("CLASSIFICATION-ONLY TEST")
        print(f"{'='*70}")
        print(f"Results: {results_yaml}")
        print(f"Expected: {expected_yaml}")
        print(f"Output: {output_dir}")
        print(f"{'='*70}\n")
    
    def normalize_label(self, label: str) -> str:
        """Clean and normalize labels for comparison"""
        # Remove common prefixes
        label = label.lower().strip()
        label = label.replace('a photo of', '')
        label = label.replace('a ', '')
        label = label.replace('an ', '')
        label = label.strip()
        
        # Remove extra spaces
        label = ' '.join(label.split())
        
        return label
    
    def labels_match(self, detected: str, expected: str, fuzzy: bool = True) -> tuple:
        """
        Check if labels match with different levels of strictness
        
        Returns:
            (exact_match: bool, fuzzy_match: bool, similarity_score: float)
        """
        detected_norm = self.normalize_label(detected)
        expected_norm = self.normalize_label(expected)
        
        # Exact match
        exact = (detected_norm == expected_norm)
        
        # Fuzzy match: one contains the other
        fuzzy_match = (detected_norm in expected_norm or expected_norm in detected_norm)
        
        # Word-based similarity
        detected_words = set(detected_norm.split())
        expected_words = set(expected_norm.split())
        
        if len(detected_words) == 0 or len(expected_words) == 0:
            similarity = 0.0
        else:
            # Jaccard similarity
            intersection = detected_words & expected_words
            union = detected_words | expected_words
            similarity = len(intersection) / len(union) if union else 0.0
        
        return exact, fuzzy_match, similarity
    
    def get_object_counts(self):
        """Count expected vs detected objects by type"""
        expected_counts = defaultdict(int)
        detected_counts = defaultdict(int)
        
        # Count expected objects
        for obj_id, obj_data in self.expected['expected_objects'].items():
            label = self.normalize_label(obj_data['name'])
            count = obj_data.get('count', 1)
            expected_counts[label] += count
        
        # Count detected objects
        if 'objects' in self.results:
            for obj_id, obj_data in self.results['objects'].items():
                label = self.normalize_label(obj_data['name'])
                detected_counts[label] += 1
        
        return expected_counts, detected_counts
    
    def run_classification_test(self):
        """Run the classification test"""
        print("="*70)
        print("RUNNING CLASSIFICATION TEST")
        print("="*70)
        
        expected_counts, detected_counts = self.get_object_counts()
        
        # Get all unique labels
        all_labels = set(list(expected_counts.keys()) + list(detected_counts.keys()))
        
        print(f"\nExpected objects: {sum(expected_counts.values())}")
        print(f"Detected objects: {sum(detected_counts.values())}")
        print(f"\nPer-class comparison:\n")
        
        # Detailed comparison
        comparison_results = []
        total_correct = 0
        total_expected = 0
        total_detected = 0
        
        print(f"{'Object Type':<25} {'Expected':>10} {'Detected':>10} {'Correct':>10} {'Status':>15}")
        print("-"*70)
        
        for label in sorted(all_labels):
            expected = expected_counts.get(label, 0)
            detected = detected_counts.get(label, 0)
            correct = min(expected, detected)
            
            total_expected += expected
            total_detected += detected
            total_correct += correct
            
            # Determine status
            if expected == detected:
                status = "✓ Perfect"
            elif detected == 0:
                status = "✗ Missing"
            elif detected < expected:
                status = "⚠ Under-detected"
            elif detected > expected:
                status = "⚠ Over-detected"
            else:
                status = "?"
            
            print(f"{label:<25} {expected:>10} {detected:>10} {correct:>10} {status:>15}")
            
            comparison_results.append({
                'label': label,
                'expected': expected,
                'detected': detected,
                'correct': correct,
                'missing': max(0, expected - detected),
                'extra': max(0, detected - expected),
                'status': status
            })
        
        print("-"*70)
        print(f"{'TOTAL':<25} {total_expected:>10} {total_detected:>10} {total_correct:>10}")
        print()
        
        # Calculate metrics
        precision = total_correct / total_detected if total_detected > 0 else 0
        recall = total_correct / total_expected if total_expected > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = total_correct / max(total_expected, total_detected) if max(total_expected, total_detected) > 0 else 0
        
        self.test_results.update({
            'total_expected': total_expected,
            'total_detected': total_detected,
            'correct_labels': total_correct,
            'missing_objects': total_expected - total_correct,
            'extra_detections': total_detected - total_correct,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'per_class_results': comparison_results
        })
        
        return self.test_results
    
    def analyze_misclassifications(self):
        """Detailed analysis of what was misclassified"""
        print("\n" + "="*70)
        print("MISCLASSIFICATION ANALYSIS")
        print("="*70)
        
        if 'objects' not in self.results:
            print("No objects detected!")
            return
        
        detected_objects = self.results['objects']
        expected_counts, _ = self.get_object_counts()
        
        # Track which expected objects we've matched
        matched_expected = defaultdict(int)
        
        misclassifications = []
        correct_classifications = []
        
        for obj_id, obj_data in detected_objects.items():
            detected_label = self.normalize_label(obj_data['name'])
            confidence = obj_data.get('confidence', 0.0)
            
            # Find best match with expected objects
            best_match = None
            best_similarity = 0.0
            
            for expected_label in expected_counts.keys():
                exact, fuzzy, similarity = self.labels_match(detected_label, expected_label)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = expected_label
            
            # Check if this is correct
            if best_match and best_similarity > 0.5:  # Reasonable match threshold
                if matched_expected[best_match] < expected_counts[best_match]:
                    # Correct classification
                    matched_expected[best_match] += 1
                    correct_classifications.append({
                        'object_id': obj_id,
                        'detected': detected_label,
                        'expected': best_match,
                        'confidence': confidence,
                        'similarity': best_similarity
                    })
                else:
                    # Extra detection (already have enough of this type)
                    misclassifications.append({
                        'object_id': obj_id,
                        'detected': detected_label,
                        'issue': 'Extra detection',
                        'confidence': confidence
                    })
            else:
                # Misclassification
                misclassifications.append({
                    'object_id': obj_id,
                    'detected': detected_label,
                    'expected_closest': best_match if best_match else 'Unknown',
                    'issue': 'Wrong label',
                    'confidence': confidence,
                    'similarity': best_similarity
                })
        
        # Find missing objects
        missing = []
        for expected_label, expected_count in expected_counts.items():
            matched = matched_expected[expected_label]
            if matched < expected_count:
                missing.append({
                    'expected': expected_label,
                    'expected_count': expected_count,
                    'detected_count': matched,
                    'missing_count': expected_count - matched
                })
        
        # Print results
        print(f"\n✓ Correct Classifications: {len(correct_classifications)}")
        for item in correct_classifications:
            print(f"  • {item['detected']} (confidence: {item['confidence']:.2f})")
        
        if misclassifications:
            print(f"\n✗ Misclassifications: {len(misclassifications)}")
            for item in misclassifications:
                if item['issue'] == 'Wrong label':
                    print(f"  • Detected '{item['detected']}' (conf: {item['confidence']:.2f}) "
                          f"- Expected '{item['expected_closest']}' (similarity: {item['similarity']:.2f})")
                else:
                    print(f"  • {item['issue']}: {item['detected']} (conf: {item['confidence']:.2f})")
        
        if missing:
            print(f"\n⚠ Missing Objects: {sum(m['missing_count'] for m in missing)}")
            for item in missing:
                print(f"  • {item['expected']}: Expected {item['expected_count']}, "
                      f"detected {item['detected_count']} (missing {item['missing_count']})")
        
        print()
        
        return {
            'correct': correct_classifications,
            'misclassifications': misclassifications,
            'missing': missing
        }
    
    def plot_confusion_matrix(self):
        """Create confusion matrix visualization"""
        expected_counts, detected_counts = self.get_object_counts()
        all_labels = sorted(set(list(expected_counts.keys()) + list(detected_counts.keys())))
        
        if len(all_labels) == 0:
            print("[WARN] No objects to plot")
            return
        
        # Build confusion-like matrix (expected vs detected counts)
        matrix = np.zeros((len(all_labels), 3))  # Expected, Detected, Correct
        
        for i, label in enumerate(all_labels):
            expected = expected_counts.get(label, 0)
            detected = detected_counts.get(label, 0)
            correct = min(expected, detected)
            
            matrix[i, 0] = expected
            matrix[i, 1] = detected
            matrix[i, 2] = correct
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(all_labels) * 0.5)))
        
        x = np.arange(len(all_labels))
        width = 0.25
        
        ax.barh(x - width, matrix[:, 0], width, label='Expected', color='skyblue', edgecolor='black')
        ax.barh(x, matrix[:, 1], width, label='Detected', color='lightcoral', edgecolor='black')
        ax.barh(x + width, matrix[:, 2], width, label='Correct', color='lightgreen', edgecolor='black')
        
        ax.set_yticks(x)
        ax.set_yticklabels(all_labels)
        ax.set_xlabel('Count', fontsize=12)
        ax.set_title('Classification Test Results', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        filepath = self.output_dir / 'classification_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] {filepath}")
    
    def generate_report(self):
        """Generate comprehensive test report"""
        lines = []
        
        lines.append("="*70)
        lines.append("CLASSIFICATION TEST REPORT")
        lines.append("="*70)
        lines.append(f"Timestamp: {self.test_results['timestamp']}")
        lines.append(f"Results file: {self.results_path}")
        lines.append(f"Expected objects file: {self.expected_path}")
        lines.append("="*70)
        lines.append("")
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-"*70)
        lines.append(f"Total expected objects: {self.test_results['total_expected']}")
        lines.append(f"Total detected objects: {self.test_results['total_detected']}")
        lines.append(f"Correctly labeled: {self.test_results['correct_labels']}")
        lines.append(f"Missing objects: {self.test_results['missing_objects']}")
        lines.append(f"Extra detections: {self.test_results['extra_detections']}")
        lines.append("")
        
        # Metrics
        lines.append("METRICS")
        lines.append("-"*70)
        lines.append(f"Precision: {self.test_results['precision']:.3f} ({self.test_results['precision']*100:.1f}%)")
        lines.append(f"Recall: {self.test_results['recall']:.3f} ({self.test_results['recall']*100:.1f}%)")
        lines.append(f"F1-Score: {self.test_results['f1_score']:.3f}")
        lines.append(f"Accuracy: {self.test_results['accuracy']:.3f} ({self.test_results['accuracy']*100:.1f}%)")
        lines.append("")
        
        # Per-class results
        lines.append("PER-CLASS RESULTS")
        lines.append("-"*70)
        lines.append(f"{'Object Type':<25} {'Expected':>10} {'Detected':>10} {'Correct':>10} {'Status':>15}")
        lines.append("-"*70)
        
        for result in self.test_results['per_class_results']:
            lines.append(f"{result['label']:<25} {result['expected']:>10} {result['detected']:>10} "
                        f"{result['correct']:>10} {result['status']:>15}")
        
        lines.append("="*70)
        
        # Verdict
        lines.append("")
        lines.append("VERDICT")
        lines.append("-"*70)
        
        accuracy = self.test_results['accuracy']
        if accuracy >= 0.95:
            verdict = "✓ EXCELLENT - Nearly perfect classification"
        elif accuracy >= 0.85:
            verdict = "✓ GOOD - Reliable classification with minor errors"
        elif accuracy >= 0.70:
            verdict = "⚠ ACCEPTABLE - Noticeable misclassifications present"
        else:
            verdict = "✗ POOR - Significant classification issues"
        
        lines.append(verdict)
        lines.append("="*70)
        
        report_text = '\n'.join(lines)
        
        # Print to console
        print("\n" + report_text)
        
        # Save to file
        filepath = self.output_dir / 'classification_test_report.txt'
        with open(filepath, 'w') as f:
            f.write(report_text)
        print(f"\n[SAVED] {filepath}")
        
        return report_text
    
    def save_results(self):
        """Save detailed results as JSON"""
        filepath = self.output_dir / 'classification_test_results.json'
        with open(filepath, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"[SAVED] {filepath}")
    
    def run_complete_test(self):
        """Run all test components"""
        print("\n" + "="*70)
        print("STARTING CLASSIFICATION TEST")
        print("="*70)
        
        # Main test
        self.run_classification_test()
        
        # Detailed analysis
        misclass_analysis = self.analyze_misclassifications()
        self.test_results['misclassification_analysis'] = misclass_analysis
        
        # Visualizations
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        self.plot_confusion_matrix()
        
        # Reports
        print("\n" + "="*70)
        print("GENERATING REPORTS")
        print("="*70)
        self.save_results()
        self.generate_report()
        
        print("\n" + "="*70)
        print("✅ CLASSIFICATION TEST COMPLETE")
        print("="*70)
        print(f"Results saved to: {self.output_dir}")
        print("="*70)


def create_test_config():
    """Create template configuration file"""
    config = {
        'test_name': 'Classification Test',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'description': 'List the objects you expect to find in the scene',
        'expected_objects': {
            'blue_dice': {
                'name': 'blue dice',
                'count': 2,
                'notes': 'Two blue six-sided dice'
            },
            'red_dice': {
                'name': 'red dice',
                'count': 1,
                'notes': 'One red die'
            },
            'yellow_dice': {
                'name': 'yellow dice',
                'count': 1,
                'notes': 'One yellow die'
            }
        },
        'notes': [
            'Use exact names as they appear in your CLIP_LABELS',
            'count: how many of each object you expect',
            'The system will check if CLIP correctly identifies all objects'
        ]
    }
    
    with open('test_config.yaml', 'w') as f:
        yaml.dump(config, f, indent=2, sort_keys=False)
    
    print("Created test_config.yaml")
    print("\nEdit this file to list YOUR expected objects:")
    print("  - Set 'name' to match your object types")
    print("  - Set 'count' to how many of each you have")
    print("\nExample:")
    print("  blue_dice:")
    print("    name: 'blue dice'")
    print("    count: 2")


def main():
    parser = argparse.ArgumentParser(
        description="Classification-Only Test - Check if CLIP labels objects correctly"
    )
    parser.add_argument('--results', type=str,
                       help='Path to objects.yaml from your vision system')
    parser.add_argument('--expected-objects', type=str,
                       help='Path to test config with expected objects')
    parser.add_argument('--output', type=str, default='classification_test',
                       help='Output directory for test results')
    parser.add_argument('--create-config', action='store_true',
                       help='Create a test configuration template and exit')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_test_config()
        return
    
    if not args.results or not args.expected_objects:
        parser.error("--results and --expected-objects are required (unless using --create-config)")
    
    # Run test
    tester = ClassificationTester(
        results_yaml=args.results,
        expected_yaml=args.expected_objects,
        output_dir=args.output
    )
    
    tester.run_complete_test()


if __name__ == "__main__":
    main()