"""
PrNet Evaluation Script
=======================
Runs the pre-trained PrNet models on RouteR (rural) and RouteU (urban) test data,
generates evaluation charts and a summary report.
"""

import os
import sys
import math
import torch
import pandas as pd
import numpy as np
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Neural_Pseudorange_Correction'))

from ReadingRawGnssDataset import readingRawGnssDataset
from PrNet_parallel import MlpFeatureExtractor, PrNet
from DataLoader_SingleFile_NoTime import GNSSSingleDataFileLoader
from DataPreprocessing_PrNet_parallel import data_preprocessing


def evaluate_gnss_net_no_display(net, data_iter, batch_size, device):
    """Evaluate a model - version without d2l.Animator to avoid IPython issues."""
    net.to(device)
    net.eval()
    loss = nn.MSELoss()
    time_step = 0
    output_seq = []
    loss_values = []
    import time
    time_sum = []

    for batch in data_iter:
        time_step += 1
        x, y = [z.to(device) for z in batch]
        enc_x = x
        post_enc_x, valid_prn_index = data_preprocessing(enc_x, device)

        start_time = time.time()
        total_prm_error = net(post_enc_x)
        end_time = time.time()
        time_sum.append(end_time - start_time)

        broadcast_index_valid = valid_prn_index.unsqueeze(-1)
        J = loss(
            (total_prm_error - torch.bmm(post_enc_x[:, :, 20:21].permute(0, 2, 1), total_prm_error))[broadcast_index_valid],
            post_enc_x[:, :, 16:17][broadcast_index_valid]
        )
        loss_values.append(J.item())

        prmResi = total_prm_error - torch.bmm(post_enc_x[:, :, 20:21].permute(0, 2, 1), total_prm_error)
        for i in range(batch_size):
            enc_x_per_batch = enc_x[i]
            index_prn_prmbias_per_batch = torch.cat([
                enc_x_per_batch[valid_prn_index[i], 0:2],
                total_prm_error[i, valid_prn_index[i], :],
                enc_x_per_batch[valid_prn_index[i], 31:32],
                enc_x_per_batch[valid_prn_index[i], 34:35],
                prmResi[i, valid_prn_index[i], :]
            ], dim=1)
            output_seq.append(index_prn_prmbias_per_batch)

    if time_sum:
        elapsed_time_per_sample = sum(time_sum) / len(time_sum)
        print(f'    Inference time per sample: {elapsed_time_per_sample:.6f}s')
    print(f'    Final loss: {loss_values[-1]:.6f}' if loss_values else '    No loss computed')
    return torch.cat(output_seq, dim=0), loss_values

torch.set_default_tensor_type(torch.DoubleTensor)

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(weight_path, input_size_debiasing=16, num_hiddens=40, num_layers=20):
    """Load a pre-trained PrNet model."""
    extractor = MlpFeatureExtractor(input_size_debiasing, num_hiddens, num_layers, dropout=0)
    model = PrNet(extractor)
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model loaded from {weight_path}")
    print(f"  Total trainable parameters: {total_params}")
    return model, total_params


def load_and_prepare_data(data_file, input_size, PRN_size=32, res_size=1, label_size=3):
    """Load and prepare test data."""
    data = pd.read_csv(data_file)
    inputs = torch.tensor(data.iloc[:, 0:input_size].values)
    outputs = torch.tensor(data.iloc[:, 14:input_size].values)
    x_features, res_labels, y_labels = readingRawGnssDataset(
        inputs, outputs, input_size, res_size, label_size, PRN_size
    )
    data_iter = GNSSSingleDataFileLoader(x_features, res_labels, batch_size=1)
    return x_features, res_labels, y_labels, data_iter, data


def evaluate_single_file(model, data_file, input_size, route_name, file_label):
    """Evaluate a single test file and return results."""
    print(f"\n  Evaluating: {file_label}")
    x_features, res_labels, y_labels, data_iter, raw_data = load_and_prepare_data(
        data_file, input_size
    )

    # Run evaluation
    prm_bias, loss_values = evaluate_gnss_net_no_display(model, data_iter, 1, torch.device('cpu'))

    # Save predicted pseudorange errors
    prm_bias_np = prm_bias.cpu().detach().numpy()
    output_csv = os.path.join(OUTPUT_DIR, f'PrM_Bias_{file_label}.csv')
    pd.DataFrame(prm_bias_np).to_csv(output_csv, header=False, index=False)
    print(f"  Saved predictions to {output_csv}")

    # Compute ground truth trajectory
    gt_lon = []
    gt_lat = []
    wls_lon = []
    wls_lat = []
    for y in y_labels:
        valid_mask = y[:, 0] != 0
        if valid_mask.sum() > 0:
            gt_lon.append((y[valid_mask, 0].sum() / valid_mask.sum()).item())
            gt_lat.append((y[valid_mask, 1].sum() / valid_mask.sum()).item())

    # Extract WLS positions from features
    for x in x_features:
        valid_mask = x[:, 1] != 0
        if valid_mask.sum() > 0:
            # WLS longitude: columns 22-24 (degree, minute, second)
            lon_deg = x[valid_mask, 22].mean().item()
            lon_min = x[valid_mask, 23].mean().item()
            lon_sec = x[valid_mask, 24].mean().item()
            lat_deg = x[valid_mask, 25].mean().item()
            lat_min = x[valid_mask, 26].mean().item()
            lat_sec = x[valid_mask, 27].mean().item()
            wls_lon.append(lon_deg + lon_min / 60 + lon_sec / 3600)
            wls_lat.append(lat_deg + lat_min / 60 + lat_sec / 3600)

    # Compute pseudorange error statistics
    pr_errors = raw_data.iloc[:, 18].values  # Pseudorange Error column
    pr_errors_valid = pr_errors[~np.isnan(pr_errors)]

    results = {
        'file_label': file_label,
        'route': route_name,
        'num_epochs': len(y_labels),
        'num_samples': len(raw_data),
        'gt_lon': gt_lon,
        'gt_lat': gt_lat,
        'wls_lon': wls_lon,
        'wls_lat': wls_lat,
        'pr_errors': pr_errors_valid,
        'prm_bias': prm_bias_np,
        'loss_values': loss_values,
        'predictions_file': output_csv,
    }
    return results


def plot_trajectory(results_list, route_name, output_path):
    """Plot ground truth vs WLS trajectories."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, res in enumerate(results_list):
        color = colors[idx % len(colors)]
        if len(res['gt_lon']) > 0:
            ax.plot(res['gt_lon'], res['gt_lat'], '-', color=color, linewidth=1.5,
                    label=f"GT - {res['file_label']}", alpha=0.8)
        if len(res['wls_lon']) > 0:
            ax.plot(res['wls_lon'], res['wls_lat'], 'x', color=color, markersize=3,
                    label=f"WLS - {res['file_label']}", alpha=0.4)

    ax.set_xlabel('Longitude (degrees)', fontsize=12)
    ax.set_ylabel('Latitude (degrees)', fontsize=12)
    ax.set_title(f'{route_name} - Ground Truth vs WLS Trajectories', fontsize=14)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved trajectory plot: {output_path}")


def plot_pseudorange_errors(results_list, route_name, output_path):
    """Plot pseudorange error distributions."""
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for idx, res in enumerate(results_list):
        ax = axes[idx]
        errors = res['pr_errors']
        ax.hist(errors, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
        ax.axvline(x=np.mean(errors), color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean: {np.mean(errors):.2f} m')
        ax.axvline(x=np.median(errors), color='green', linestyle='--', linewidth=1.5,
                   label=f'Median: {np.median(errors):.2f} m')
        ax.set_xlabel('Pseudorange Error (m)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{res["file_label"]}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{route_name} - Pseudorange Error Distribution', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved error distribution plot: {output_path}")


def plot_predicted_corrections(results_list, route_name, output_path):
    """Plot PrNet predicted corrections over time."""
    n = len(results_list)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for idx, res in enumerate(results_list):
        ax = axes[idx]
        bias = res['prm_bias']
        # Column indices: 0=epoch, 1=PRN, 2=predicted_bias, ...
        if bias.shape[1] >= 3:
            # Plot predicted bias for each sample
            ax.scatter(range(len(bias)), bias[:, 2], s=1, alpha=0.5, c='steelblue')
            ax.set_xlabel('Sample Index', fontsize=11)
            ax.set_ylabel('Predicted Pseudorange Correction (m)', fontsize=11)
            ax.set_title(f'{res["file_label"]}', fontsize=12)
            ax.grid(True, alpha=0.3)

            # Stats
            mean_corr = np.mean(bias[:, 2])
            std_corr = np.std(bias[:, 2])
            ax.axhline(y=mean_corr, color='red', linestyle='--', linewidth=1,
                        label=f'Mean: {mean_corr:.2f} m, Std: {std_corr:.2f} m')
            ax.legend(fontsize=9)

    fig.suptitle(f'{route_name} - PrNet Predicted Corrections', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved corrections plot: {output_path}")


def plot_error_per_epoch(results_list, route_name, output_path):
    """Plot pseudorange error statistics per epoch."""
    n = len(results_list)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for idx, res in enumerate(results_list):
        ax = axes[idx]
        bias = res['prm_bias']
        if bias.shape[1] >= 3:
            # Group by epoch (column 0)
            epochs = np.unique(bias[:, 0])
            mean_per_epoch = []
            for ep in epochs:
                mask = bias[:, 0] == ep
                mean_per_epoch.append(np.mean(np.abs(bias[mask, 2])))
            ax.plot(range(len(mean_per_epoch)), mean_per_epoch, '-', linewidth=1, color='steelblue')
            ax.set_xlabel('Time Step', fontsize=11)
            ax.set_ylabel('Mean |Predicted Correction| (m)', fontsize=11)
            ax.set_title(f'{res["file_label"]}', fontsize=12)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f'{route_name} - Mean Absolute Correction per Time Step', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved per-epoch error plot: {output_path}")


def plot_evaluation_loss(results_list, route_name, output_path):
    """Plot evaluation loss over time steps."""
    n = len(results_list)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for idx, res in enumerate(results_list):
        ax = axes[idx]
        losses = res.get('loss_values', [])
        if losses:
            ax.plot(range(1, len(losses) + 1), losses, '-', linewidth=1, color='coral')
            ax.set_xlabel('Time Step', fontsize=11)
            ax.set_ylabel('MSE Loss', fontsize=11)
            ax.set_title(f'{res["file_label"]} (Mean Loss: {np.mean(losses):.4f})', fontsize=12)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f'{route_name} - Evaluation Loss per Time Step', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved evaluation loss plot: {output_path}")


def generate_report(all_results, model_params):
    """Generate a markdown summary report."""
    report_lines = [
        "# PrNet Evaluation Report",
        "",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Model Architecture",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Model | PrNet (MLP-based) |",
        f"| Input Features | 16 (CN0, sinE, cosE, PRN, WLS positions, geometry vectors, heading) |",
        f"| Hidden Neurons | 40 |",
        f"| MLP Layers | 20 |",
        f"| Total Trainable Parameters | {model_params:,} |",
        f"| Dropout | 0 |",
        "",
        "## Evaluation Results",
        "",
    ]

    for route_name, results_list in all_results.items():
        report_lines.append(f"### {route_name}")
        report_lines.append("")
        report_lines.append("| Test File | # Epochs | # Samples | PR Error Mean (m) | PR Error Std (m) | PR Error Median (m) |")
        report_lines.append("|-----------|----------|-----------|-------------------|------------------|---------------------|")
        for res in results_list:
            pr_mean = np.mean(res['pr_errors'])
            pr_std = np.std(res['pr_errors'])
            pr_median = np.median(res['pr_errors'])
            report_lines.append(
                f"| {res['file_label']} | {res['num_epochs']} | {res['num_samples']} | "
                f"{pr_mean:.4f} | {pr_std:.4f} | {pr_median:.4f} |"
            )
        report_lines.append("")

        # Add correction stats
        report_lines.append("**PrNet Predicted Corrections:**")
        report_lines.append("")
        report_lines.append("| Test File | Mean Correction (m) | Std Correction (m) | Min (m) | Max (m) |")
        report_lines.append("|-----------|--------------------|--------------------|---------|---------|")
        for res in results_list:
            bias = res['prm_bias']
            if bias.shape[1] >= 3:
                corr = bias[:, 2]
                report_lines.append(
                    f"| {res['file_label']} | {np.mean(corr):.4f} | {np.std(corr):.4f} | "
                    f"{np.min(corr):.4f} | {np.max(corr):.4f} |"
                )
        report_lines.append("")

    report_lines.extend([
        "## Generated Charts",
        "",
        "### RouteR (Rural Area)",
        "",
        "**Trajectory Plot:**",
        "",
        "![RouteR Trajectory](RouteR_trajectory.png)",
        "",
        "**Pseudorange Error Distribution:**",
        "",
        "![RouteR PR Error Distribution](RouteR_pr_error_distribution.png)",
        "",
        "**PrNet Predicted Corrections:**",
        "",
        "![RouteR Predicted Corrections](RouteR_predicted_corrections.png)",
        "",
        "**Mean Absolute Correction per Time Step:**",
        "",
        "![RouteR Error per Epoch](RouteR_error_per_epoch.png)",
        "",
        "**Evaluation Loss per Time Step:**",
        "",
        "![RouteR Evaluation Loss](RouteR_evaluation_loss.png)",
        "",
        "### RouteU (Urban Area)",
        "",
        "**Trajectory Plot:**",
        "",
        "![RouteU Trajectory](RouteU_trajectory.png)",
        "",
        "**Pseudorange Error Distribution:**",
        "",
        "![RouteU PR Error Distribution](RouteU_pr_error_distribution.png)",
        "",
        "**PrNet Predicted Corrections:**",
        "",
        "![RouteU Predicted Corrections](RouteU_predicted_corrections.png)",
        "",
        "**Mean Absolute Correction per Time Step:**",
        "",
        "![RouteU Error per Epoch](RouteU_error_per_epoch.png)",
        "",
        "**Evaluation Loss per Time Step:**",
        "",
        "![RouteU Evaluation Loss](RouteU_evaluation_loss.png)",
        "",
        "## Notes",
        "",
        "- Rural data uses `input_size=39`, Urban data uses `input_size=55`",
        "- Evaluation uses batch_size=1 (one epoch per batch)",
        "- Pre-trained weights from `Neural_Pseudorange_Correction/Weights/` are used",
        "- All evaluations run on CPU",
        "",
    ])

    report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\n  Report saved to {report_path}")
    return report_path


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    all_results = {}
    model_params = 0

    # ==================== RouteR (Rural) ====================
    print("=" * 60)
    print("RouteR (Rural Area) Evaluation")
    print("=" * 60)

    weight_r = os.path.join(base_dir, 'Neural_Pseudorange_Correction', 'Weights', 'RouteR',
                            'PrNet_Layer20_H40_heading_RouteR_500.tar')
    model_r, model_params = load_model(weight_r)

    route_r_test_dir = os.path.join(base_dir, 'Data', 'RouteR', 'Testing')
    route_r_test_files = [
        f for f in os.listdir(route_r_test_dir)
        if f.endswith('.csv') and not os.path.isdir(os.path.join(route_r_test_dir, f))
    ]
    route_r_test_files.sort()

    route_r_results = []
    for test_file in route_r_test_files:
        file_path = os.path.join(route_r_test_dir, test_file)
        file_label = test_file.replace('SvPVT3D_Error_label_dynamic_', '').replace('.csv', '')
        try:
            res = evaluate_single_file(model_r, file_path, input_size=39, route_name='RouteR',
                                       file_label=file_label)
            route_r_results.append(res)
        except Exception as e:
            print(f"  ERROR evaluating {file_label}: {e}")

    if route_r_results:
        all_results['RouteR (Rural)'] = route_r_results
        print("\n  Generating RouteR charts...")
        plot_trajectory(route_r_results, 'RouteR (Rural)',
                       os.path.join(OUTPUT_DIR, 'RouteR_trajectory.png'))
        plot_pseudorange_errors(route_r_results, 'RouteR (Rural)',
                               os.path.join(OUTPUT_DIR, 'RouteR_pr_error_distribution.png'))
        plot_predicted_corrections(route_r_results, 'RouteR (Rural)',
                                   os.path.join(OUTPUT_DIR, 'RouteR_predicted_corrections.png'))
        plot_error_per_epoch(route_r_results, 'RouteR (Rural)',
                            os.path.join(OUTPUT_DIR, 'RouteR_error_per_epoch.png'))
        plot_evaluation_loss(route_r_results, 'RouteR (Rural)',
                            os.path.join(OUTPUT_DIR, 'RouteR_evaluation_loss.png'))

    # ==================== RouteU (Urban) ====================
    print("\n" + "=" * 60)
    print("RouteU (Urban Area) Evaluation")
    print("=" * 60)

    weight_u = os.path.join(base_dir, 'Neural_Pseudorange_Correction', 'Weights', 'RouteU',
                            'PrNet_Layer20_H40_heading_RouteUS5_2000.tar')
    model_u, _ = load_model(weight_u)

    route_u_test_dir = os.path.join(base_dir, 'Data', 'RouteU', 'Testing')
    route_u_test_files = [
        f for f in os.listdir(route_u_test_dir)
        if f.endswith('.csv') and not os.path.isdir(os.path.join(route_u_test_dir, f))
    ]
    route_u_test_files.sort()

    route_u_results = []
    for test_file in route_u_test_files:
        file_path = os.path.join(route_u_test_dir, test_file)
        file_label = test_file.replace('SvPVT3D_Error_label_dynamic_', '').replace('.csv', '')
        try:
            res = evaluate_single_file(model_u, file_path, input_size=55, route_name='RouteU',
                                       file_label=file_label)
            route_u_results.append(res)
        except Exception as e:
            print(f"  ERROR evaluating {file_label}: {e}")

    if route_u_results:
        all_results['RouteU (Urban)'] = route_u_results
        print("\n  Generating RouteU charts...")
        plot_trajectory(route_u_results, 'RouteU (Urban)',
                       os.path.join(OUTPUT_DIR, 'RouteU_trajectory.png'))
        plot_pseudorange_errors(route_u_results, 'RouteU (Urban)',
                               os.path.join(OUTPUT_DIR, 'RouteU_pr_error_distribution.png'))
        plot_predicted_corrections(route_u_results, 'RouteU (Urban)',
                                   os.path.join(OUTPUT_DIR, 'RouteU_predicted_corrections.png'))
        plot_error_per_epoch(route_u_results, 'RouteU (Urban)',
                            os.path.join(OUTPUT_DIR, 'RouteU_error_per_epoch.png'))
        plot_evaluation_loss(route_u_results, 'RouteU (Urban)',
                            os.path.join(OUTPUT_DIR, 'RouteU_evaluation_loss.png'))

    # ==================== Generate Report ====================
    print("\n" + "=" * 60)
    print("Generating Summary Report")
    print("=" * 60)
    generate_report(all_results, model_params)

    print("\n" + "=" * 60)
    print("DONE! All results saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
