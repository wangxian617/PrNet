"""
Chart Generation for PrNet Experiments
=======================================
Generates all required plots for pseudorange data, position data,
ablation experiments, and computational overhead comparison.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker


# Color palette for methods
METHOD_COLORS = {
    'WLS': '#1f77b4',
    'WLS+EKF': '#ff7f0e',
    'WLS+Corr': '#2ca02c',
    'WLS+EKF+RTS': '#d62728',
    'WLS+MHE': '#9467bd',
    'WLS+PrNet': '#e377c2',
    'WLS+PrNet+RTS': '#7f7f7f',
    'WLS+EKF+PrNet': '#bcbd22',
    'WLS+MHE+PrNet': '#17becf',
    'Ground Truth': '#000000',
}


def _setup_plot_style():
    """Set up consistent plotting style."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })


# ============================================================
# 1. Raw Pseudorange Time Series per Satellite
# ============================================================

def plot_raw_pseudorange_timeseries(epoch_data_dict, sorted_epochs, scenario_name, output_dir):
    """Plot raw pseudorange values per satellite over time."""
    _setup_plot_style()

    # Collect per-PRN data
    prn_data = {}
    for ep in sorted_epochs:
        ed = epoch_data_dict[ep]
        for i, prn in enumerate(ed['prns']):
            if prn not in prn_data:
                prn_data[prn] = {'epochs': [], 'pr': []}
            prn_data[prn]['epochs'].append(ep)
            prn_data[prn]['pr'].append(ed['raw_pr'][i])

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.tab20
    prn_list = sorted(prn_data.keys())

    for idx, prn in enumerate(prn_list):
        color = cmap(idx / max(len(prn_list), 1))
        ax.plot(prn_data[prn]['epochs'], prn_data[prn]['pr'],
                '.', markersize=2, color=color, alpha=0.7, label=f'PRN {prn}')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Raw Pseudorange (m)')
    ax.set_title(f'{scenario_name} - Raw Pseudorange per Satellite')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, f'{scenario_name}_raw_pseudorange_timeseries.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 2. Pseudorange Error Distribution Histogram (multi-method)
# ============================================================

def plot_pr_error_distribution_comparison(method_pr_errors, scenario_name, output_dir):
    """
    Plot pseudorange error distribution for multiple methods.
    method_pr_errors: dict of method_name -> array of PR errors
    """
    _setup_plot_style()

    n_methods = len(method_pr_errors)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for idx, (method, errors) in enumerate(method_pr_errors.items()):
        ax = axes[idx]
        if len(errors) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(method)
            continue

        color = METHOD_COLORS.get(method, '#333333')
        ax.hist(errors, bins=60, color=color, edgecolor='white', alpha=0.7)
        ax.axvline(np.mean(errors), color='red', ls='--', lw=1.5,
                   label=f'Mean: {np.mean(errors):.2f} m')
        ax.axvline(np.median(errors), color='green', ls='--', lw=1.5,
                   label=f'Median: {np.median(errors):.2f} m')
        ax.set_xlabel('Pseudorange Error (m)')
        ax.set_ylabel('Count')
        ax.set_title(method)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{scenario_name} - Pseudorange Error Distribution', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{scenario_name}_pr_error_distribution.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 3. Pseudorange Error CDF Curves
# ============================================================

def plot_pr_error_cdf(method_pr_errors, scenario_name, output_dir):
    """Plot CDF of pseudorange errors for multiple methods."""
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, errors in method_pr_errors.items():
        if len(errors) == 0:
            continue
        sorted_err = np.sort(np.abs(errors))
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        color = METHOD_COLORS.get(method, '#333333')
        ax.plot(sorted_err, cdf, label=method, color=color, linewidth=1.5)

    ax.set_xlabel('|Pseudorange Error| (m)')
    ax.set_ylabel('CDF')
    ax.set_title(f'{scenario_name} - Pseudorange Error CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    path = os.path.join(output_dir, f'{scenario_name}_pr_error_cdf.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 4. Pseudorange Error Box Plot
# ============================================================

def plot_pr_error_boxplot(method_pr_errors, scenario_name, output_dir):
    """Box plot of pseudorange errors across methods."""
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    data = []
    for method, errors in method_pr_errors.items():
        if len(errors) > 0:
            labels.append(method)
            data.append(errors)

    if not data:
        plt.close()
        return None

    bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    colors = [METHOD_COLORS.get(l, '#cccccc') for l in labels]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Pseudorange Error (m)')
    ax.set_title(f'{scenario_name} - Pseudorange Error Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    path = os.path.join(output_dir, f'{scenario_name}_pr_error_boxplot.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 5. Trajectory Comparison Plot
# ============================================================

def plot_trajectory_comparison(method_positions, gt_positions, scenario_name, output_dir):
    """
    Plot trajectories for multiple methods vs ground truth.
    method_positions: dict of method_name -> list of (lat, lon)
    gt_positions: list of (lat, lon) for ground truth
    """
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 9))

    if gt_positions is not None and len(gt_positions) > 0:
        gt = np.array(gt_positions)
        ax.plot(gt[:, 1], gt[:, 0], 'k-', linewidth=2, label='Ground Truth', zorder=10)

    for method, positions in method_positions.items():
        if len(positions) == 0:
            continue
        pos = np.array(positions)
        color = METHOD_COLORS.get(method, '#333333')
        ax.plot(pos[:, 1], pos[:, 0], '.', markersize=3, color=color, alpha=0.5, label=method)

    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title(f'{scenario_name} - Trajectory Comparison')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{scenario_name}_trajectory_comparison.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 6. Position Error Time Series
# ============================================================

def plot_position_error_timeseries(method_errors, scenario_name, output_dir):
    """
    Plot position error over time for multiple methods.
    method_errors: dict of method_name -> (epochs_array, errors_array)
    """
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    for method, (epochs, errors) in method_errors.items():
        if len(errors) == 0:
            continue
        color = METHOD_COLORS.get(method, '#333333')
        ax.plot(epochs, errors, '-', linewidth=0.8, color=color, alpha=0.7, label=method)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Horizontal Position Error (m)')
    ax.set_title(f'{scenario_name} - Position Error Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{scenario_name}_position_error_timeseries.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 7. Position Error CDF
# ============================================================

def plot_position_error_cdf(method_errors, scenario_name, output_dir):
    """Plot CDF of position errors for multiple methods."""
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, (epochs, errors) in method_errors.items():
        if len(errors) == 0:
            continue
        sorted_err = np.sort(errors)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        color = METHOD_COLORS.get(method, '#333333')
        ax.plot(sorted_err, cdf, label=method, color=color, linewidth=1.5)

    ax.set_xlabel('Horizontal Position Error (m)')
    ax.set_ylabel('CDF')
    ax.set_title(f'{scenario_name} - Position Error CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{scenario_name}_position_error_cdf.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 8. Position Error Statistics Table (as image)
# ============================================================

def plot_position_error_table(method_errors, scenario_name, output_dir):
    """Generate a table image of position error statistics."""
    _setup_plot_style()

    headers = ['Method', 'Mean (m)', 'Median (m)', '50th %', '67th %', '95th %', 'Std (m)']
    rows = []

    for method, (epochs, errors) in method_errors.items():
        if len(errors) == 0:
            rows.append([method, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
            continue
        rows.append([
            method,
            f'{np.mean(errors):.2f}',
            f'{np.median(errors):.2f}',
            f'{np.percentile(errors, 50):.2f}',
            f'{np.percentile(errors, 67):.2f}',
            f'{np.percentile(errors, 95):.2f}',
            f'{np.std(errors):.2f}',
        ])

    fig, ax = plt.subplots(figsize=(12, max(2, 0.5 * len(rows) + 1.5)))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', weight='bold')

    ax.set_title(f'{scenario_name} - Position Error Statistics', fontsize=13, pad=20)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{scenario_name}_position_error_table.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 9. Ablation Experiment Bar Chart
# ============================================================

def plot_ablation_results(ablation_results, baseline_error, scenario_name, output_dir):
    """
    Bar chart comparing ablation experiment results.
    ablation_results: dict of ablation_type -> {name, avg_loss, ...}
    baseline_error: float, the full PrNet average loss for comparison
    """
    _setup_plot_style()

    names = ['Full PrNet'] + [r['name'] for r in ablation_results.values()]
    losses = [baseline_error] + [r['avg_loss'] for r in ablation_results.values()]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2ca02c'] + ['#ff7f0e'] * len(ablation_results)
    bars = ax.bar(range(len(names)), losses, color=colors, edgecolor='white', alpha=0.8)

    # Add value labels
    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Average MSE Loss')
    ax.set_title(f'{scenario_name} - Feature Ablation Study')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(output_dir, f'{scenario_name}_ablation_bar.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 10. Ablation Experiment Table
# ============================================================

def plot_ablation_table(ablation_results, baseline_stats, scenario_name, output_dir):
    """Generate ablation experiment statistics table as image."""
    _setup_plot_style()

    headers = ['Configuration', 'Input Size', '# Params', 'Avg Loss', 'Train Time (s)', 'Inference (ms)']
    rows = [['Full PrNet (baseline)', '16', str(baseline_stats.get('num_params', 'N/A')),
             f"{baseline_stats.get('avg_loss', 0):.6f}",
             f"{baseline_stats.get('train_time', 0):.1f}",
             f"{baseline_stats.get('avg_inference_time', 0) * 1000:.2f}"]]

    for abl_type, res in ablation_results.items():
        rows.append([
            res['name'],
            str(res['input_size']),
            str(res['num_params']),
            f"{res['avg_loss']:.6f}",
            f"{res['train_time']:.1f}",
            f"{res['avg_inference_time'] * 1000:.2f}",
        ])

    fig, ax = plt.subplots(figsize=(14, max(2, 0.5 * len(rows) + 1.5)))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', weight='bold')

    ax.set_title(f'{scenario_name} - Feature Ablation Statistics', fontsize=13, pad=20)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{scenario_name}_ablation_table.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 11. Computational Overhead Comparison
# ============================================================

def plot_computational_overhead(model_stats, output_dir):
    """
    Bar chart comparing computational overhead of different models.
    model_stats: dict of model_name -> {params, inference_time, train_time}
    """
    _setup_plot_style()

    models = list(model_stats.keys())
    params = [model_stats[m].get('params', 0) for m in models]
    inference_times = [model_stats[m].get('inference_time', 0) * 1000 for m in models]  # ms

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Parameters
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000'][:len(models)]
    bars1 = ax1.bar(models, params, color=colors, edgecolor='white', alpha=0.8)
    for bar, val in zip(bars1, params):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{val:,}', ha='center', va='bottom', fontsize=9)
    ax1.set_ylabel('Number of Parameters')
    ax1.set_title('Model Complexity')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20, ha='right')

    # Inference time
    bars2 = ax2.bar(models, inference_times, color=colors, edgecolor='white', alpha=0.8)
    for bar, val in zip(bars2, inference_times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('Inference Time (ms/sample)')
    ax2.set_title('Inference Speed')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=20, ha='right')

    fig.suptitle('Computational Overhead Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'computational_overhead_comparison.png')
    plt.savefig(path)
    plt.close()
    return path


def plot_computational_overhead_table(model_stats, output_dir):
    """Generate computational overhead table as image."""
    _setup_plot_style()

    headers = ['Model', '# Parameters', 'Inference (ms)', 'Training (s)', 'Memory Est. (MB)']
    rows = []
    for model, stats in model_stats.items():
        mem_mb = stats.get('params', 0) * 8 / (1024 * 1024)  # 8 bytes per double param
        rows.append([
            model,
            f"{stats.get('params', 0):,}",
            f"{stats.get('inference_time', 0) * 1000:.3f}",
            f"{stats.get('train_time', 0):.1f}",
            f"{mem_mb:.2f}",
        ])

    fig, ax = plt.subplots(figsize=(12, max(2, 0.5 * len(rows) + 1.5)))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', weight='bold')

    ax.set_title('Computational Overhead Summary', fontsize=13, pad=20)
    plt.tight_layout()
    path = os.path.join(output_dir, 'computational_overhead_table.png')
    plt.savefig(path)
    plt.close()
    return path
