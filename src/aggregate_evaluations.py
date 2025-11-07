#!/usr/bin/env python3
"""
Aggregate Multiple Evaluation Runs for Statistical Analysis

This script combines results from multiple evaluation runs to generate
mean ± std statistics for your capstone report.

Usage:
    python aggregate_evaluations.py --input eval_run_* --output aggregated_results
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import yaml


class EvaluationAggregator:
    """Aggregate multiple evaluation runs"""
    
    def __init__(self, evaluation_dirs: List[str], output_dir: str = "aggregated_results"):
        self.eval_dirs = [Path(d) for d in evaluation_dirs]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.all_metrics = []
        self.load_all_metrics()
    
    def load_all_metrics(self):
        """Load metrics from all evaluation runs"""
        for eval_dir in self.eval_dirs:
            metrics_file = eval_dir / 'evaluation_metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    self.all_metrics.append(json.load(f))
                print(f"[LOADED] {metrics_file}")
            else:
                print(f"[WARNING] Not found: {metrics_file}")
        
        print(f"\n[INFO] Loaded {len(self.all_metrics)} evaluation runs\n")
    
    def aggregate_detection_metrics(self) -> Dict:
        """Aggregate detection metrics"""
        precisions = [m['detection']['precision'] for m in self.all_metrics]
        recalls = [m['detection']['recall'] for m in self.all_metrics]
        f1_scores = [m['detection']['f1_score'] for m in self.all_metrics]
        maps = [m['detection']['mAP'] for m in self.all_metrics]
        
        return {
            'precision': {
                'mean': np.mean(precisions),
                'std': np.std(precisions),
                'min': np.min(precisions),
                'max': np.max(precisions),
                'all_values': precisions
            },
            'recall': {
                'mean': np.mean(recalls),
                'std': np.std(recalls),
                'min': np.min(recalls),
                'max': np.max(recalls),
                'all_values': recalls
            },
            'f1_score': {
                'mean': np.mean(f1_scores),
                'std': np.std(f1_scores),
                'min': np.min(f1_scores),
                'max': np.max(f1_scores),
                'all_values': f1_scores
            },
            'mAP': {
                'mean': np.mean(maps),
                'std': np.std(maps),
                'min': np.min(maps),
                'max': np.max(maps),
                'all_values': maps
            }
        }
    
    def aggregate_classification_metrics(self) -> Dict:
        """Aggregate classification metrics"""
        accuracies = [m['classification']['accuracy'] for m in self.all_metrics]
        confidences = [m['classification']['mean_confidence'] for m in self.all_metrics]
        
        return {
            'accuracy': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies),
                'all_values': accuracies
            },
            'mean_confidence': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
                'all_values': confidences
            }
        }
    
    def aggregate_localization_metrics(self) -> Dict:
        """Aggregate localization metrics"""
        mean_errors = []
        x_errors = []
        y_errors = []
        z_errors = []
        
        for m in self.all_metrics:
            if 'mean_error_mm' in m['localization']:
                mean_errors.append(m['localization']['mean_error_mm'])
                x_errors.append(m['localization']['per_axis']['x_mean_mm'])
                y_errors.append(m['localization']['per_axis']['y_mean_mm'])
                z_errors.append(m['localization']['per_axis']['z_mean_mm'])
        
        if not mean_errors:
            return {}
        
        return {
            'position_error_mm': {
                'mean': np.mean(mean_errors),
                'std': np.std(mean_errors),
                'min': np.min(mean_errors),
                'max': np.max(mean_errors),
                'all_values': mean_errors
            },
            'x_error_mm': {
                'mean': np.mean(x_errors),
                'std': np.std(x_errors),
                'all_values': x_errors
            },
            'y_error_mm': {
                'mean': np.mean(y_errors),
                'std': np.std(y_errors),
                'all_values': y_errors
            },
            'z_error_mm': {
                'mean': np.mean(z_errors),
                'std': np.std(z_errors),
                'all_values': z_errors
            }
        }
    
    def aggregate_dimension_metrics(self) -> Dict:
        """Aggregate dimension metrics"""
        length_errors = []
        width_errors = []
        height_errors = []
        
        for m in self.all_metrics:
            dims = m.get('dimensions', {})
            if 'length_mean_error_mm' in dims:
                length_errors.append(dims['length_mean_error_mm'])
                width_errors.append(dims['width_mean_error_mm'])
                height_errors.append(dims['height_mean_error_mm'])
        
        if not length_errors:
            return {}
        
        return {
            'length_error_mm': {
                'mean': np.mean(length_errors),
                'std': np.std(length_errors),
                'all_values': length_errors
            },
            'width_error_mm': {
                'mean': np.mean(width_errors),
                'std': np.std(width_errors),
                'all_values': width_errors
            },
            'height_error_mm': {
                'mean': np.mean(height_errors),
                'std': np.std(height_errors),
                'all_values': height_errors
            }
        }
    
    def plot_aggregated_results(self, aggregated: Dict):
        """Create visualizations of aggregated results"""
        
        # 1. Detection metrics bar chart
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Precision
        det = aggregated['detection']
        metrics_names = ['Precision', 'Recall', 'F1-Score', 'mAP@0.10m']
        metrics_data = [det['precision'], det['recall'], det['f1_score'], det['mAP']]
        
        for ax, name, data in zip(axes.flat, metrics_names, metrics_data):
            ax.bar(range(len(data['all_values'])), data['all_values'], alpha=0.7, edgecolor='black')
            ax.axhline(data['mean'], color='r', linestyle='--', linewidth=2, label=f"Mean: {data['mean']:.3f}")
            ax.axhline(data['mean'] + data['std'], color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.axhline(data['mean'] - data['std'], color='orange', linestyle=':', linewidth=1.5, alpha=0.7, 
                      label=f"Std: ±{data['std']:.3f}")
            ax.set_xlabel('Run Number')
            ax.set_ylabel(name)
            ax.set_title(f'{name} Across Runs')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / 'detection_metrics_aggregated.png'
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[SAVED] {filepath}")
        
        # 2. Position error distribution
        if 'position_error_mm' in aggregated['localization']:
            loc = aggregated['localization']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram
            all_errors = loc['position_error_mm']['all_values']
            ax1.hist(all_errors, bins=15, alpha=0.7, edgecolor='black')
            ax1.axvline(loc['position_error_mm']['mean'], color='r', linestyle='--', linewidth=2,
                       label=f"Mean: {loc['position_error_mm']['mean']:.2f} mm")
            ax1.set_xlabel('Position Error (mm)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Aggregated Position Error Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Per-axis comparison
            axes_names = ['X', 'Y', 'Z']
            axes_data = [loc['x_error_mm'], loc['y_error_mm'], loc['z_error_mm']]
            means = [d['mean'] for d in axes_data]
            stds = [d['std'] for d in axes_data]
            
            x_pos = np.arange(len(axes_names))
            ax2.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Axis')
            ax2.set_ylabel('Mean Error (mm)')
            ax2.set_title('Position Error by Axis (Aggregated)')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(axes_names)
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            filepath = self.output_dir / 'localization_aggregated.png'
            plt.savefig(filepath, dpi=300)
            plt.close()
            print(f"[SAVED] {filepath}")
        
        # 3. Classification accuracy over runs
        cls = aggregated['classification']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        runs = range(1, len(cls['accuracy']['all_values']) + 1)
        ax.plot(runs, cls['accuracy']['all_values'], 'o-', linewidth=2, markersize=8, label='Accuracy')
        ax.axhline(cls['accuracy']['mean'], color='r', linestyle='--', linewidth=2,
                  label=f"Mean: {cls['accuracy']['mean']:.3f}")
        ax.fill_between(runs, 
                        cls['accuracy']['mean'] - cls['accuracy']['std'],
                        cls['accuracy']['mean'] + cls['accuracy']['std'],
                        alpha=0.2, color='red', label=f"±1 Std")
        ax.set_xlabel('Run Number')
        ax.set_ylabel('Classification Accuracy')
        ax.set_title('Classification Accuracy Across Runs')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        filepath = self.output_dir / 'classification_aggregated.png'
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[SAVED] {filepath}")
    
    def generate_latex_table(self, aggregated: Dict):
        """Generate LaTeX table for aggregated results"""
        latex = []
        
        latex.append("% Aggregated Detection Performance")
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Aggregated Object Detection Performance (N=" + str(len(self.all_metrics)) + " runs)}")
        latex.append("\\begin{tabular}{lcc}")
        latex.append("\\hline")
        latex.append("Metric & Mean $\\pm$ Std & Range \\\\")
        latex.append("\\hline")
        
        det = aggregated['detection']
        latex.append(f"Precision & {det['precision']['mean']:.3f} $\\pm$ {det['precision']['std']:.3f} & "
                    f"[{det['precision']['min']:.3f}, {det['precision']['max']:.3f}] \\\\")
        latex.append(f"Recall & {det['recall']['mean']:.3f} $\\pm$ {det['recall']['std']:.3f} & "
                    f"[{det['recall']['min']:.3f}, {det['recall']['max']:.3f}] \\\\")
        latex.append(f"F1-Score & {det['f1_score']['mean']:.3f} $\\pm$ {det['f1_score']['std']:.3f} & "
                    f"[{det['f1_score']['min']:.3f}, {det['f1_score']['max']:.3f}] \\\\")
        latex.append(f"mAP@0.10m & {det['mAP']['mean']:.3f} $\\pm$ {det['mAP']['std']:.3f} & "
                    f"[{det['mAP']['min']:.3f}, {det['mAP']['max']:.3f}] \\\\")
        
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\label{tab:detection_aggregated}")
        latex.append("\\end{table}")
        latex.append("")
        
        # Localization table
        if 'position_error_mm' in aggregated['localization']:
            latex.append("% Aggregated Localization Performance")
            latex.append("\\begin{table}[h]")
            latex.append("\\centering")
            latex.append("\\caption{Aggregated 3D Localization Accuracy (N=" + str(len(self.all_metrics)) + " runs)}")
            latex.append("\\begin{tabular}{lc}")
            latex.append("\\hline")
            latex.append("Metric & Mean $\\pm$ Std (mm) \\\\")
            latex.append("\\hline")
            
            loc = aggregated['localization']
            latex.append(f"Position Error & {loc['position_error_mm']['mean']:.2f} $\\pm$ {loc['position_error_mm']['std']:.2f} \\\\")
            latex.append(f"X-axis Error & {loc['x_error_mm']['mean']:.2f} $\\pm$ {loc['x_error_mm']['std']:.2f} \\\\")
            latex.append(f"Y-axis Error & {loc['y_error_mm']['mean']:.2f} $\\pm$ {loc['y_error_mm']['std']:.2f} \\\\")
            latex.append(f"Z-axis Error & {loc['z_error_mm']['mean']:.2f} $\\pm$ {loc['z_error_mm']['std']:.2f} \\\\")
            
            latex.append("\\hline")
            latex.append("\\end{tabular}")
            latex.append("\\label{tab:localization_aggregated}")
            latex.append("\\end{table}")
        
        filepath = self.output_dir / 'aggregated_latex_tables.tex'
        with open(filepath, 'w') as f:
            f.write('\n'.join(latex))
        print(f"[SAVED] {filepath}")
    
    def generate_report(self, aggregated: Dict):
        """Generate comprehensive text report"""
        lines = []
        
        lines.append("="*70)
        lines.append("AGGREGATED EVALUATION REPORT")
        lines.append("="*70)
        lines.append(f"Number of runs: {len(self.all_metrics)}")
        lines.append(f"Evaluation directories: {', '.join([str(d) for d in self.eval_dirs])}")
        lines.append("="*70)
        lines.append("")
        
        # Detection
        lines.append("1. DETECTION PERFORMANCE")
        lines.append("-"*70)
        det = aggregated['detection']
        for metric_name in ['precision', 'recall', 'f1_score', 'mAP']:
            data = det[metric_name]
            metric_display = metric_name.replace('_', ' ').title()
            if metric_name == 'mAP':
                metric_display = 'mAP@0.10m'
            lines.append(f"   {metric_display}:")
            lines.append(f"     Mean ± Std: {data['mean']:.3f} ± {data['std']:.3f}")
            lines.append(f"     Range: [{data['min']:.3f}, {data['max']:.3f}]")
            lines.append(f"     Percentage: {data['mean']*100:.1f}% ± {data['std']*100:.1f}%")
        lines.append("")
        
        # Classification
        lines.append("2. CLASSIFICATION PERFORMANCE")
        lines.append("-"*70)
        cls = aggregated['classification']
        lines.append(f"   Accuracy:")
        lines.append(f"     Mean ± Std: {cls['accuracy']['mean']:.3f} ± {cls['accuracy']['std']:.3f}")
        lines.append(f"     Percentage: {cls['accuracy']['mean']*100:.1f}% ± {cls['accuracy']['std']*100:.1f}%")
        lines.append(f"   Mean Confidence:")
        lines.append(f"     Mean ± Std: {cls['mean_confidence']['mean']:.3f} ± {cls['mean_confidence']['std']:.3f}")
        lines.append("")
        
        # Localization
        if 'position_error_mm' in aggregated['localization']:
            lines.append("3. LOCALIZATION ACCURACY")
            lines.append("-"*70)
            loc = aggregated['localization']
            lines.append(f"   Position Error (mm):")
            lines.append(f"     Mean ± Std: {loc['position_error_mm']['mean']:.2f} ± {loc['position_error_mm']['std']:.2f}")
            lines.append(f"     Range: [{loc['position_error_mm']['min']:.2f}, {loc['position_error_mm']['max']:.2f}]")
            lines.append(f"   Per-axis Errors (mm):")
            lines.append(f"     X: {loc['x_error_mm']['mean']:.2f} ± {loc['x_error_mm']['std']:.2f}")
            lines.append(f"     Y: {loc['y_error_mm']['mean']:.2f} ± {loc['y_error_mm']['std']:.2f}")
            lines.append(f"     Z: {loc['z_error_mm']['mean']:.2f} ± {loc['z_error_mm']['std']:.2f}")
            lines.append("")
        
        # Dimensions
        if aggregated['dimensions']:
            lines.append("4. DIMENSION ESTIMATION")
            lines.append("-"*70)
            dims = aggregated['dimensions']
            for dim in ['length', 'width', 'height']:
                key = f'{dim}_error_mm'
                if key in dims:
                    lines.append(f"   {dim.capitalize()} Error: {dims[key]['mean']:.2f} ± {dims[key]['std']:.2f} mm")
            lines.append("")
        
        lines.append("="*70)
        
        report_text = '\n'.join(lines)
        print("\n" + report_text)
        
        filepath = self.output_dir / 'aggregated_report.txt'
        with open(filepath, 'w') as f:
            f.write(report_text)
        print(f"\n[SAVED] {filepath}")
    
    def run_aggregation(self):
        """Run complete aggregation"""
        print("\n" + "="*70)
        print("AGGREGATING EVALUATION RESULTS")
        print("="*70)
        
        aggregated = {
            'detection': self.aggregate_detection_metrics(),
            'classification': self.aggregate_classification_metrics(),
            'localization': self.aggregate_localization_metrics(),
            'dimensions': self.aggregate_dimension_metrics()
        }
        
        # Save aggregated metrics
        filepath = self.output_dir / 'aggregated_metrics.json'
        with open(filepath, 'w') as f:
            json.dump(aggregated, f, indent=2)
        print(f"[SAVED] {filepath}")
        
        # Generate visualizations
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        self.plot_aggregated_results(aggregated)
        
        # Generate reports
        print("\n" + "="*70)
        print("GENERATING REPORTS")
        print("="*70)
        self.generate_latex_table(aggregated)
        self.generate_report(aggregated)
        
        print("\n" + "="*70)
        print("✅ AGGREGATION COMPLETE")
        print("="*70)
        print(f"All outputs saved to: {self.output_dir}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Aggregate Multiple Evaluation Runs")
    parser.add_argument('--input', nargs='+', required=True,
                       help='Paths to evaluation directories (can use wildcards)')
    parser.add_argument('--output', type=str, default='aggregated_results',
                       help='Output directory for aggregated results')
    
    args = parser.parse_args()
    
    aggregator = EvaluationAggregator(args.input, args.output)
    aggregator.run_aggregation()


if __name__ == "__main__":
    main()
