#!/usr/bin/env python3
"""
Comprehensive PrNet Experiment Runner
=====================================
Runs all experiments:
1. Extract raw pseudorange data per satellite
2. WLS/EKF/RTS/MHE positioning from pre-processed CSV data
3. PrNet evaluation (using pre-trained weights)
4. PrNet + EKF/RTS/MHE combined positioning
5. Comparison models (PBC-RF, FCNN-LSTM, Set Transformer)
6. Feature ablation experiments (6 classes)
7. Computational overhead comparison
8. Generate all charts and report

Usage:
    cd PrNet
    python run_all_experiments.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from torch import nn

# Ensure module paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'Neural_Pseudorange_Correction'))
sys.path.insert(0, os.path.join(BASE_DIR, 'experiments'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ReadingRawGnssDataset import readingRawGnssDataset
from PrNet_parallel import MlpFeatureExtractor, PrNet
from DataLoader_SingleFile_NoTime import GNSSSingleDataFileLoader
from DataPreprocessing_PrNet_parallel import data_preprocessing

from gnss_positioning import (
    extract_epoch_data, parse_csv_columns,
    wls_position_from_csv, run_ekf, run_rts_smoothing, run_mhe,
    apply_prnet_corrections, compute_position_errors,
    xyz2lla, compute_horizontal_error_m
)
from charts import (
    plot_raw_pseudorange_timeseries,
    plot_pr_error_distribution_comparison,
    plot_pr_error_cdf,
    plot_pr_error_boxplot,
    plot_trajectory_comparison,
    plot_position_error_timeseries,
    plot_position_error_cdf,
    plot_position_error_table,
    plot_ablation_results,
    plot_ablation_table,
    plot_computational_overhead,
    plot_computational_overhead_table,
)

torch.set_default_tensor_type(torch.DoubleTensor)

# Output directory
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Data Loading
# ============================================================

def load_csv_data(filepath):
    """Load a pre-processed CSV file and extract epoch data."""
    with open(filepath, 'r') as f:
        header = f.readline().strip()
    col_map = parse_csv_columns(header)
    data = pd.read_csv(filepath).values
    epoch_data = extract_epoch_data(data, col_map)
    sorted_epochs = sorted(epoch_data.keys())
    return epoch_data, sorted_epochs, col_map, data


def load_prnet_model(weight_path, input_size=16, num_hiddens=40, num_layers=20):
    """Load a pre-trained PrNet model."""
    extractor = MlpFeatureExtractor(input_size, num_hiddens, num_layers, dropout=0)
    model = PrNet(extractor)
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, total_params


def prepare_prnet_data(filepath, input_size, prn_size=32):
    """Prepare data for PrNet evaluation."""
    data = pd.read_csv(filepath)
    inputs = torch.tensor(data.iloc[:, 0:input_size].values)
    outputs = torch.tensor(data.iloc[:, 14:input_size].values)
    x_features, res_labels, y_labels = readingRawGnssDataset(
        inputs, outputs, input_size, 1, 3, prn_size
    )
    data_iter = GNSSSingleDataFileLoader(x_features, res_labels, batch_size=1)
    return x_features, res_labels, y_labels, data_iter


def evaluate_prnet(model, data_iter, device=torch.device('cpu')):
    """Evaluate PrNet model and return predictions and timing."""
    model.to(device)
    model.eval()
    loss_fn = nn.MSELoss()
    output_seq = []
    loss_values = []
    inference_times = []

    with torch.no_grad():
        for batch in data_iter:
            x, y = [z.to(device) for z in batch]
            post_x, valid_idx = data_preprocessing(x, device)

            start = time.time()
            preds = model(post_x)
            inference_times.append(time.time() - start)

            broadcast_idx = valid_idx.unsqueeze(-1)
            J = loss_fn(
                (preds - torch.bmm(post_x[:, :, 20:21].permute(0, 2, 1), preds))[broadcast_idx],
                post_x[:, :, 16:17][broadcast_idx]
            )
            loss_values.append(J.item())

            prmResi = preds - torch.bmm(post_x[:, :, 20:21].permute(0, 2, 1), preds)
            for i in range(x.shape[0]):
                mask = valid_idx[i]
                enc_x = x[i]
                out = torch.cat([
                    enc_x[mask, 0:2],
                    preds[i, mask, :],
                    enc_x[mask, 31:32],
                    enc_x[mask, 34:35],
                    prmResi[i, mask, :]
                ], dim=1)
                output_seq.append(out.cpu().numpy())

    if output_seq:
        predictions = np.vstack(output_seq)
    else:
        predictions = np.array([]).reshape(0, 7)

    avg_inference = np.mean(inference_times) if inference_times else 0
    return predictions, loss_values, avg_inference


# ============================================================
# Per-scenario experiment pipeline
# ============================================================

def run_scenario_experiments(filepath, scenario_name, model, input_size,
                             training_files=None, route_type='R'):
    """Run all experiments for a single test scenario."""
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*60}")

    scenario_dir = os.path.join(OUTPUT_DIR, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    # ---- Load data ----
    print("  Loading data...")
    epoch_data, sorted_epochs, col_map, raw_data = load_csv_data(filepath)
    print(f"  {len(sorted_epochs)} epochs, {len(raw_data)} total samples")

    # ---- Phase 1: Raw pseudorange extraction ----
    print("  Phase 1: Extracting raw pseudorange data...")
    raw_pr_records = []
    for ep in sorted_epochs:
        ed = epoch_data[ep]
        for i, prn in enumerate(ed['prns']):
            raw_pr_records.append({
                'Epoch': ep, 'PRN': prn,
                'Raw_Pseudorange': ed['raw_pr'][i],
                'CN0': ed['cn0'][i],
                'Elevation': ed['elevation'][i],
            })
    raw_pr_df = pd.DataFrame(raw_pr_records)
    raw_pr_path = os.path.join(scenario_dir, f'raw_pseudorange_per_sv_{scenario_name}.csv')
    raw_pr_df.to_csv(raw_pr_path, index=False)
    print(f"  Saved: {raw_pr_path}")

    plot_raw_pseudorange_timeseries(epoch_data, sorted_epochs, scenario_name, scenario_dir)

    # ---- Phase 2: WLS positioning ----
    print("  Phase 2: WLS positioning...")
    wls_results = {}
    wls_pr_errors = []
    for ep in sorted_epochs:
        ed = epoch_data[ep]
        wls_xyz, wls_lla, pr_res = wls_position_from_csv(ed)
        wls_results[ep] = {'xyz': wls_xyz, 'lla': wls_lla, 'pr_residuals': pr_res}
        wls_pr_errors.extend(pr_res.tolist())

    # ---- Phase 2b: EKF ----
    print("  Phase 2b: EKF positioning...")
    ekf_results = run_ekf(epoch_data, sorted_epochs)
    ekf_pr_errors = []
    for ep in sorted_epochs:
        if ep in ekf_results and len(ekf_results[ep].get('pr_residuals', [])) > 0:
            ekf_pr_errors.extend(ekf_results[ep]['pr_residuals'].tolist())

    # ---- Phase 2c: EKF + RTS ----
    print("  Phase 2c: EKF + RTS smoothing...")
    rts_results = run_rts_smoothing(ekf_results, sorted_epochs)

    # ---- Phase 2d: MHE ----
    print("  Phase 2d: MHE positioning...")
    mhe_results = run_mhe(epoch_data, sorted_epochs)
    mhe_pr_errors = []
    for ep in sorted_epochs:
        if ep in mhe_results and len(mhe_results[ep].get('pr_residuals', [])) > 0:
            mhe_pr_errors.extend(mhe_results[ep]['pr_residuals'].tolist())

    # ---- Phase 3: PrNet evaluation ----
    print("  Phase 3: PrNet evaluation...")
    x_features, res_labels, y_labels, data_iter = prepare_prnet_data(filepath, input_size)
    prnet_predictions, prnet_losses, prnet_avg_inference = evaluate_prnet(model, data_iter)
    prnet_avg_loss = np.mean(prnet_losses) if prnet_losses else 0

    # Save PrNet predictions
    prnet_csv = os.path.join(scenario_dir, f'PrM_Bias_{scenario_name}.csv')
    pd.DataFrame(prnet_predictions).to_csv(prnet_csv, header=False, index=False)

    # ---- Phase 3b: WLS + PrNet ----
    print("  Phase 3b: WLS + PrNet positioning...")
    corrected_epoch_data = apply_prnet_corrections(epoch_data, prnet_predictions, sorted_epochs)
    prnet_wls_results = {}
    prnet_pr_errors = []
    for ep in sorted_epochs:
        ed = corrected_epoch_data[ep]
        wls_xyz, wls_lla, pr_res = wls_position_from_csv(ed)
        prnet_wls_results[ep] = {'xyz': wls_xyz, 'lla': wls_lla, 'pr_residuals': pr_res}
        prnet_pr_errors.extend(pr_res.tolist())

    # ---- Phase 3c: WLS + PrNet + RTS ----
    print("  Phase 3c: EKF + PrNet...")
    ekf_prnet_results = run_ekf(corrected_epoch_data, sorted_epochs)

    print("  Phase 3d: PrNet + RTS...")
    prnet_rts_results = run_rts_smoothing(ekf_prnet_results, sorted_epochs)

    # ---- Phase 3e: MHE + PrNet ----
    print("  Phase 3e: MHE + PrNet...")
    mhe_prnet_results = run_mhe(corrected_epoch_data, sorted_epochs)

    # ---- Save position data ----
    print("  Saving position data...")
    all_methods = {
        'WLS': wls_results,
        'WLS+EKF': ekf_results,
        'WLS+EKF+RTS': rts_results,
        'WLS+MHE': mhe_results,
        'WLS+PrNet': prnet_wls_results,
        'WLS+EKF+PrNet': ekf_prnet_results,
        'WLS+PrNet+RTS': prnet_rts_results,
        'WLS+MHE+PrNet': mhe_prnet_results,
    }

    for method_name, results in all_methods.items():
        pos_records = []
        for ep in sorted_epochs:
            if ep in results:
                r = results[ep]
                pos_records.append({
                    'Epoch': ep,
                    'Lat': r['lla'][0], 'Lon': r['lla'][1], 'Alt': r['lla'][2],
                })
        method_safe = method_name.replace('+', '_').replace(' ', '_')
        pos_df = pd.DataFrame(pos_records)
        pos_path = os.path.join(scenario_dir, f'{method_safe}_position_{scenario_name}.csv')
        pos_df.to_csv(pos_path, index=False)

    # ---- Save pseudorange data per method ----
    for method_name, results in all_methods.items():
        pr_records = []
        for ep in sorted_epochs:
            if ep in results and ep in epoch_data:
                r = results[ep]
                ed = epoch_data[ep]
                pr_res = r.get('pr_residuals', np.array([]))
                if len(pr_res) > 0 and len(pr_res) == len(ed['prns']):
                    for i, prn in enumerate(ed['prns']):
                        pr_records.append({
                            'Epoch': ep, 'PRN': prn,
                            'PR_Residual': pr_res[i],
                        })
        method_safe = method_name.replace('+', '_').replace(' ', '_')
        pr_df = pd.DataFrame(pr_records)
        pr_path = os.path.join(scenario_dir, f'{method_safe}_pseudorange_{scenario_name}.csv')
        pr_df.to_csv(pr_path, index=False)

    # ---- Compute position errors ----
    print("  Computing position errors...")
    method_pos_errors = {}
    for method_name, results in all_methods.items():
        epochs_arr, errors_arr = compute_position_errors(results, epoch_data, sorted_epochs)
        method_pos_errors[method_name] = (epochs_arr, errors_arr)

    # ---- Generate charts ----
    print("  Generating charts...")

    # PR error distributions
    method_pr_errors_dict = {
        'WLS': np.array(wls_pr_errors),
        'WLS+EKF': np.array(ekf_pr_errors),
        'WLS+MHE': np.array(mhe_pr_errors),
        'WLS+PrNet': np.array(prnet_pr_errors),
    }
    plot_pr_error_distribution_comparison(method_pr_errors_dict, scenario_name, scenario_dir)
    plot_pr_error_cdf(method_pr_errors_dict, scenario_name, scenario_dir)
    plot_pr_error_boxplot(method_pr_errors_dict, scenario_name, scenario_dir)

    # Trajectories
    gt_positions = []
    method_traj = {}
    for method_name, results in all_methods.items():
        positions = []
        for ep in sorted_epochs:
            if ep in results:
                lla = results[ep]['lla']
                positions.append((lla[0], lla[1]))
        method_traj[method_name] = positions

    for ep in sorted_epochs:
        if ep in epoch_data:
            gt_lla = xyz2lla(epoch_data[ep]['gt_xyz'])
            gt_positions.append((gt_lla[0], gt_lla[1]))

    plot_trajectory_comparison(method_traj, gt_positions, scenario_name, scenario_dir)

    # Position errors
    plot_position_error_timeseries(method_pos_errors, scenario_name, scenario_dir)
    plot_position_error_cdf(method_pos_errors, scenario_name, scenario_dir)
    plot_position_error_table(method_pos_errors, scenario_name, scenario_dir)

    print(f"  Scenario {scenario_name} complete!")

    return {
        'scenario': scenario_name,
        'num_epochs': len(sorted_epochs),
        'num_samples': len(raw_data),
        'method_pos_errors': method_pos_errors,
        'method_pr_errors': method_pr_errors_dict,
        'prnet_predictions': prnet_predictions,
        'prnet_avg_loss': prnet_avg_loss,
        'prnet_avg_inference': prnet_avg_inference,
    }


# ============================================================
# Ablation Experiments
# ============================================================

def run_ablation_experiments_for_route(training_dir, test_filepath, input_size,
                                        route_name, num_epochs=50, lr=0.01):
    """Run ablation experiments for a route."""
    print(f"\n{'='*60}")
    print(f"Ablation Experiments: {route_name}")
    print(f"{'='*60}")

    from ablation import ABLATION_CONFIGS, train_ablated_model, evaluate_ablated_model
    from DataLoader_MultipleFile_Random_NoTime import GnssMultipleDataFileLoader

    abl_dir = os.path.join(OUTPUT_DIR, f'ablation_{route_name}')
    os.makedirs(abl_dir, exist_ok=True)

    # Load training data
    training_files = [
        os.path.join(training_dir, f)
        for f in sorted(os.listdir(training_dir))
        if f.endswith('.csv')
    ]

    if not training_files:
        print(f"  No training files found in {training_dir}")
        return {}

    print(f"  Loading {len(training_files)} training files...")
    train_features_list = []
    train_labels_list = []
    for tf in training_files:
        data = pd.read_csv(tf)
        inputs = torch.tensor(data.iloc[:, 0:input_size].values)
        outputs = torch.tensor(data.iloc[:, 14:input_size].values)
        x_feat, res_lab, y_lab = readingRawGnssDataset(inputs, outputs, input_size, 1, 3, 32)
        train_features_list.append(x_feat)
        train_labels_list.append(res_lab)

    def make_train_iter():
        return GnssMultipleDataFileLoader(train_features_list, train_labels_list, batch_size=32)

    # Load test data
    print(f"  Loading test file: {test_filepath}")
    test_x, test_res, test_y, test_iter = prepare_prnet_data(test_filepath, input_size)

    def make_test_iter():
        return GNSSSingleDataFileLoader(test_x, test_res, batch_size=1)

    # Also compute baseline PrNet loss for comparison
    weight_path = os.path.join(
        BASE_DIR, 'Neural_Pseudorange_Correction', 'Weights',
        f'Route{route_name}',
        os.listdir(os.path.join(BASE_DIR, 'Neural_Pseudorange_Correction', 'Weights', f'Route{route_name}'))[0]
    )
    baseline_model, baseline_params = load_prnet_model(weight_path)
    _, baseline_losses, baseline_inference = evaluate_prnet(baseline_model, make_test_iter())
    baseline_avg_loss = np.mean(baseline_losses) if baseline_losses else 0

    baseline_stats = {
        'num_params': baseline_params,
        'avg_loss': baseline_avg_loss,
        'train_time': 0,  # pre-trained
        'avg_inference_time': baseline_inference,
    }

    # Run all ablation types
    ablation_results = {}
    for abl_type, config in ABLATION_CONFIGS.items():
        print(f"\n  --- Ablation: {config['name']} ---")

        model, params, train_time = train_ablated_model(
            abl_type, make_train_iter(), num_epochs=num_epochs, lr=lr
        )

        predictions, losses, inference_times = evaluate_ablated_model(
            model, abl_type, make_test_iter()
        )

        avg_loss = np.mean(losses) if losses else 0
        avg_inf = np.mean(inference_times) if inference_times else 0

        ablation_results[abl_type] = {
            'name': config['name'],
            'description': config['description'],
            'input_size': config['input_size'],
            'num_params': params,
            'train_time': train_time,
            'avg_loss': avg_loss,
            'avg_inference_time': avg_inf,
            'predictions': predictions,
            'losses': losses,
        }

        # Save predictions
        pred_path = os.path.join(abl_dir, f'ablation_{abl_type}_predictions.csv')
        pd.DataFrame(predictions).to_csv(pred_path, header=False, index=False)

    # Generate ablation charts
    plot_ablation_results(ablation_results, baseline_avg_loss, route_name, abl_dir)
    plot_ablation_table(ablation_results, baseline_stats, route_name, abl_dir)

    return ablation_results, baseline_stats


# ============================================================
# Comparison Models
# ============================================================

def run_comparison_models(training_dir, test_filepath, input_size, route_name, num_epochs=50):
    """Run comparison models (PBC-RF, FCNN-LSTM, Set Transformer)."""
    print(f"\n{'='*60}")
    print(f"Comparison Models: {route_name}")
    print(f"{'='*60}")

    from comparison_models import PBCRF, FCNNLSTMWrapper, SetTransformerWrapper
    from DataLoader_MultipleFile_Random_NoTime import GnssMultipleDataFileLoader

    comp_dir = os.path.join(OUTPUT_DIR, f'comparison_{route_name}')
    os.makedirs(comp_dir, exist_ok=True)

    # Load training data
    training_files = [
        os.path.join(training_dir, f)
        for f in sorted(os.listdir(training_dir))
        if f.endswith('.csv')
    ]

    train_features_list = []
    train_labels_list = []
    for tf in training_files:
        data = pd.read_csv(tf)
        inputs = torch.tensor(data.iloc[:, 0:input_size].values)
        outputs = torch.tensor(data.iloc[:, 14:input_size].values)
        x_feat, res_lab, y_lab = readingRawGnssDataset(inputs, outputs, input_size, 1, 3, 32)
        train_features_list.append(x_feat)
        train_labels_list.append(res_lab)

    def make_train_iter():
        return GnssMultipleDataFileLoader(train_features_list, train_labels_list, batch_size=32)

    # Load test data
    test_x, test_res, test_y, test_iter = prepare_prnet_data(test_filepath, input_size)

    def make_test_iter():
        return GNSSSingleDataFileLoader(test_x, test_res, batch_size=1)

    model_stats = {}

    # --- PBC-RF ---
    print("\n  --- PBC-RF (Random Forest) ---")
    rf = PBCRF(n_estimators=100, max_depth=20)
    rf.train(make_train_iter())
    rf_preds, rf_inf_times = rf.predict_from_iter(make_test_iter())
    rf_avg_inf = np.mean(rf_inf_times) if rf_inf_times else 0

    pd.DataFrame(rf_preds).to_csv(os.path.join(comp_dir, 'PBC_RF_predictions.csv'),
                                   header=False, index=False)
    model_stats['PBC-RF'] = {
        'params': rf.num_parameters,
        'inference_time': rf_avg_inf,
        'train_time': rf.train_time,
        'predictions': rf_preds,
    }
    print(f"    Params: {rf.num_parameters}, Inference: {rf_avg_inf*1000:.3f}ms, Train: {rf.train_time:.1f}s")

    # --- FCNN-LSTM ---
    print("\n  --- FCNN-LSTM ---")
    fcnn = FCNNLSTMWrapper(input_size=16, fc_hidden=64, lstm_hidden=64,
                           num_fc_layers=3, num_lstm_layers=2)
    fcnn.train(make_train_iter(), num_epochs=num_epochs, lr=0.01)
    fcnn_preds, fcnn_inf_times = fcnn.evaluate(make_test_iter())
    fcnn_avg_inf = np.mean(fcnn_inf_times) if fcnn_inf_times else 0

    pd.DataFrame(fcnn_preds).to_csv(os.path.join(comp_dir, 'FCNN_LSTM_predictions.csv'),
                                     header=False, index=False)
    model_stats['FCNN-LSTM'] = {
        'params': fcnn.num_parameters,
        'inference_time': fcnn_avg_inf,
        'train_time': fcnn.train_time,
        'predictions': fcnn_preds,
    }
    print(f"    Params: {fcnn.num_parameters}, Inference: {fcnn_avg_inf*1000:.3f}ms, Train: {fcnn.train_time:.1f}s")

    # --- Set Transformer ---
    print("\n  --- Set Transformer ---")
    st = SetTransformerWrapper(input_size=16, dim_hidden=64, num_heads=4,
                               num_sab_layers=2, num_seeds=1)
    st.train(make_train_iter(), num_epochs=num_epochs, lr=0.01)
    st_preds, st_inf_times = st.evaluate(make_test_iter())
    st_avg_inf = np.mean(st_inf_times) if st_inf_times else 0

    pd.DataFrame(st_preds).to_csv(os.path.join(comp_dir, 'Set_Transformer_predictions.csv'),
                                   header=False, index=False)
    model_stats['Set Transformer'] = {
        'params': st.num_parameters,
        'inference_time': st_avg_inf,
        'train_time': st.train_time,
        'predictions': st_preds,
    }
    print(f"    Params: {st.num_parameters}, Inference: {st_avg_inf*1000:.3f}ms, Train: {st.train_time:.1f}s")

    # --- PrNet (baseline) ---
    weight_path = os.path.join(
        BASE_DIR, 'Neural_Pseudorange_Correction', 'Weights',
        f'Route{route_name}',
        os.listdir(os.path.join(BASE_DIR, 'Neural_Pseudorange_Correction', 'Weights', f'Route{route_name}'))[0]
    )
    prnet_model, prnet_params = load_prnet_model(weight_path)
    prnet_preds, prnet_losses, prnet_inf = evaluate_prnet(prnet_model, make_test_iter())
    model_stats['PrNet'] = {
        'params': prnet_params,
        'inference_time': prnet_inf,
        'train_time': 0,  # pre-trained
        'predictions': prnet_preds,
    }

    # Generate comparison charts
    plot_computational_overhead(model_stats, comp_dir)
    plot_computational_overhead_table(model_stats, comp_dir)

    return model_stats


# ============================================================
# Report Generation
# ============================================================

def generate_comprehensive_report(all_scenario_results, ablation_results_all,
                                   model_stats_all):
    """Generate a comprehensive markdown report."""
    lines = [
        "# PrNet Comprehensive Experiment Report",
        "",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Overview",
        "",
        "This report presents results from comprehensive GNSS pseudorange correction experiments",
        "covering multiple positioning methods, PrNet neural network corrections, feature ablation",
        "studies, and comparison with baseline models.",
        "",
        "### Methods Evaluated:",
        "- **WLS**: Weighted Least Squares",
        "- **WLS+EKF**: Extended Kalman Filter",
        "- **WLS+EKF+RTS**: EKF with Rauch-Tung-Striebel smoothing",
        "- **WLS+MHE**: Moving Horizon Estimator",
        "- **WLS+PrNet**: PrNet pseudorange correction",
        "- **WLS+EKF+PrNet**: EKF with PrNet corrections",
        "- **WLS+PrNet+RTS**: PrNet + RTS smoothing",
        "- **WLS+MHE+PrNet**: MHE with PrNet corrections",
        "",
        "---",
        "",
        "## 2. Per-Scenario Results",
        "",
    ]

    for scenario_result in all_scenario_results:
        name = scenario_result['scenario']
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Epochs: {scenario_result['num_epochs']}")
        lines.append(f"- Samples: {scenario_result['num_samples']}")
        lines.append(f"- PrNet Avg Loss: {scenario_result['prnet_avg_loss']:.6f}")
        lines.append(f"- PrNet Avg Inference: {scenario_result['prnet_avg_inference']*1000:.3f} ms")
        lines.append("")

        # Position error table
        lines.append("#### Position Errors")
        lines.append("")
        lines.append("| Method | Mean (m) | Median (m) | 50th % | 67th % | 95th % |")
        lines.append("|--------|----------|------------|--------|--------|--------|")

        for method, (eps, errs) in scenario_result['method_pos_errors'].items():
            if len(errs) > 0:
                lines.append(
                    f"| {method} | {np.mean(errs):.2f} | {np.median(errs):.2f} | "
                    f"{np.percentile(errs, 50):.2f} | {np.percentile(errs, 67):.2f} | "
                    f"{np.percentile(errs, 95):.2f} |"
                )
            else:
                lines.append(f"| {method} | N/A | N/A | N/A | N/A | N/A |")
        lines.append("")

        # PR error stats
        lines.append("#### Pseudorange Error Statistics")
        lines.append("")
        lines.append("| Method | Mean (m) | Std (m) | Median (m) |")
        lines.append("|--------|----------|---------|------------|")
        for method, errors in scenario_result['method_pr_errors'].items():
            if len(errors) > 0:
                lines.append(
                    f"| {method} | {np.mean(errors):.2f} | {np.std(errors):.2f} | "
                    f"{np.median(errors):.2f} |"
                )
        lines.append("")

        # Chart references
        lines.append("#### Charts")
        lines.append("")
        lines.append(f"![Raw Pseudorange]({name}/{name}_raw_pseudorange_timeseries.png)")
        lines.append(f"![Trajectory]({name}/{name}_trajectory_comparison.png)")
        lines.append(f"![PR Error Distribution]({name}/{name}_pr_error_distribution.png)")
        lines.append(f"![PR Error CDF]({name}/{name}_pr_error_cdf.png)")
        lines.append(f"![PR Error Boxplot]({name}/{name}_pr_error_boxplot.png)")
        lines.append(f"![Position Error]({name}/{name}_position_error_timeseries.png)")
        lines.append(f"![Position CDF]({name}/{name}_position_error_cdf.png)")
        lines.append(f"![Position Table]({name}/{name}_position_error_table.png)")
        lines.append("")

    # Ablation section
    lines.append("---")
    lines.append("")
    lines.append("## 3. Feature Ablation Results")
    lines.append("")

    for route_name, (abl_results, baseline) in ablation_results_all.items():
        lines.append(f"### Route {route_name}")
        lines.append("")
        lines.append(f"Baseline PrNet: params={baseline['num_params']}, avg_loss={baseline['avg_loss']:.6f}")
        lines.append("")
        lines.append("| Ablation | Input Size | Params | Avg Loss | Change vs Baseline |")
        lines.append("|----------|-----------|--------|----------|-------------------|")
        for abl_type, res in abl_results.items():
            change = ((res['avg_loss'] - baseline['avg_loss']) / max(baseline['avg_loss'], 1e-10)) * 100
            lines.append(
                f"| {res['name']} | {res['input_size']} | {res['num_params']} | "
                f"{res['avg_loss']:.6f} | {change:+.1f}% |"
            )
        lines.append("")
        lines.append(f"![Ablation Bar](ablation_{route_name}/{route_name}_ablation_bar.png)")
        lines.append(f"![Ablation Table](ablation_{route_name}/{route_name}_ablation_table.png)")
        lines.append("")

    # Computational overhead section
    lines.append("---")
    lines.append("")
    lines.append("## 4. Computational Overhead")
    lines.append("")

    for route_name, stats in model_stats_all.items():
        lines.append(f"### Route {route_name}")
        lines.append("")
        lines.append("| Model | Parameters | Inference (ms) | Training (s) |")
        lines.append("|-------|-----------|----------------|-------------|")
        for model, s in stats.items():
            lines.append(
                f"| {model} | {s['params']:,} | {s['inference_time']*1000:.3f} | {s['train_time']:.1f} |"
            )
        lines.append("")
        lines.append(f"![Overhead](comparison_{route_name}/computational_overhead_comparison.png)")
        lines.append(f"![Overhead Table](comparison_{route_name}/computational_overhead_table.png)")
        lines.append("")

    report_path = os.path.join(OUTPUT_DIR, 'comprehensive_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nReport saved to: {report_path}")
    return report_path


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("PrNet Comprehensive Experiment Pipeline")
    print("=" * 60)

    device = torch.device('cpu')

    # ---- Load models ----
    weight_r = os.path.join(BASE_DIR, 'Neural_Pseudorange_Correction', 'Weights', 'RouteR',
                            'PrNet_Layer20_H40_heading_RouteR_500.tar')
    weight_u = os.path.join(BASE_DIR, 'Neural_Pseudorange_Correction', 'Weights', 'RouteU',
                            'PrNet_Layer20_H40_heading_RouteUS5_2000.tar')

    print("\nLoading PrNet models...")
    model_r, params_r = load_prnet_model(weight_r)
    print(f"  RouteR model: {params_r:,} parameters")
    model_u, params_u = load_prnet_model(weight_u)
    print(f"  RouteU model: {params_u:,} parameters")

    # ---- Define test scenarios ----
    route_r_test_dir = os.path.join(BASE_DIR, 'Data', 'RouteR', 'Testing')
    route_u_test_dir = os.path.join(BASE_DIR, 'Data', 'RouteU', 'Testing')

    scenarios = []
    # RouteR scenarios
    for f in sorted(os.listdir(route_r_test_dir)):
        if f.endswith('.csv') and not os.path.isdir(os.path.join(route_r_test_dir, f)):
            label = f.replace('SvPVT3D_Error_label_dynamic_', '').replace('.csv', '')
            scenarios.append({
                'filepath': os.path.join(route_r_test_dir, f),
                'name': label,
                'model': model_r,
                'input_size': 39,
                'route': 'R',
            })

    # RouteU scenarios
    for f in sorted(os.listdir(route_u_test_dir)):
        if f.endswith('.csv') and not os.path.isdir(os.path.join(route_u_test_dir, f)):
            label = f.replace('SvPVT3D_Error_label_dynamic_', '').replace('.csv', '')
            scenarios.append({
                'filepath': os.path.join(route_u_test_dir, f),
                'name': label,
                'model': model_u,
                'input_size': 55,
                'route': 'U',
            })

    # ---- Run per-scenario experiments ----
    all_scenario_results = []
    for sc in scenarios:
        result = run_scenario_experiments(
            sc['filepath'], sc['name'], sc['model'], sc['input_size'],
            route_type=sc['route']
        )
        all_scenario_results.append(result)

    # ---- Ablation experiments ----
    # Use reduced epochs for faster execution
    ablation_epochs = 10
    ablation_results_all = {}

    # RouteR ablation (use first test file)
    route_r_train_dir = os.path.join(BASE_DIR, 'Data', 'RouteR', 'Training')
    if os.path.isdir(route_r_train_dir) and os.listdir(route_r_train_dir):
        r_test_file = os.path.join(route_r_test_dir,
                                    'SvPVT3D_Error_label_dynamic_2020-05-14-US-MTV-1.csv')
        if os.path.exists(r_test_file):
            abl_r = run_ablation_experiments_for_route(
                route_r_train_dir, r_test_file, 39, 'R',
                num_epochs=ablation_epochs
            )
            if abl_r:
                ablation_results_all['R'] = abl_r

    # RouteU ablation
    route_u_train_dir = os.path.join(BASE_DIR, 'Data', 'RouteU', 'Training')
    if os.path.isdir(route_u_train_dir) and os.listdir(route_u_train_dir):
        u_test_file = os.path.join(route_u_test_dir,
                                    'SvPVT3D_Error_label_dynamic_2021-04-28-US-SJC-1.csv')
        if os.path.exists(u_test_file):
            abl_u = run_ablation_experiments_for_route(
                route_u_train_dir, u_test_file, 55, 'U',
                num_epochs=ablation_epochs
            )
            if abl_u:
                ablation_results_all['U'] = abl_u

    # ---- Comparison models ----
    comp_epochs = 10
    model_stats_all = {}

    if os.path.isdir(route_r_train_dir) and os.listdir(route_r_train_dir):
        r_test_file = os.path.join(route_r_test_dir,
                                    'SvPVT3D_Error_label_dynamic_2020-05-14-US-MTV-1.csv')
        if os.path.exists(r_test_file):
            stats_r = run_comparison_models(
                route_r_train_dir, r_test_file, 39, 'R', num_epochs=comp_epochs
            )
            model_stats_all['R'] = stats_r

    if os.path.isdir(route_u_train_dir) and os.listdir(route_u_train_dir):
        u_test_file = os.path.join(route_u_test_dir,
                                    'SvPVT3D_Error_label_dynamic_2021-04-28-US-SJC-1.csv')
        if os.path.exists(u_test_file):
            stats_u = run_comparison_models(
                route_u_train_dir, u_test_file, 55, 'U', num_epochs=comp_epochs
            )
            model_stats_all['U'] = stats_u

    # ---- Generate comprehensive report ----
    print("\n" + "=" * 60)
    print("Generating Comprehensive Report")
    print("=" * 60)
    generate_comprehensive_report(all_scenario_results, ablation_results_all, model_stats_all)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Results directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
