"""
Feature Ablation Experiments for PrNet
======================================
6-class feature ablation study removing one feature group at a time.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import copy


class AblatedMlpFeatureExtractor(nn.Module):
    """MLP Feature Extractor with configurable feature ablation."""

    def __init__(self, input_size_debiasing, num_hiddens_debiasing,
                 num_debiasing_layers, dropout=0, ablation_type=None, **kwargs):
        super().__init__(**kwargs)
        self.num_debiasing_layers = num_debiasing_layers
        self.ablation_type = ablation_type

        self.debiasing_mlp_0 = nn.Sequential(
            nn.Linear(input_size_debiasing, num_hiddens_debiasing),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        if num_debiasing_layers >= 2:
            self.debiasing_mlp = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(num_hiddens_debiasing, num_hiddens_debiasing),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ) for _ in range(num_debiasing_layers - 1)
            ])
        self.debiasing_mlp_output = nn.Linear(num_hiddens_debiasing, 1)

    def forward(self, inputs, *args):
        X = inputs
        for i in range(self.num_debiasing_layers):
            if i == 0:
                h_temp = self.debiasing_mlp_0(X)
            else:
                h = h_temp
                h_temp = self.debiasing_mlp[i - 1](h)
        prm_error_bias = self.debiasing_mlp_output(h_temp)
        return prm_error_bias


class AblatedPrNet(nn.Module):
    """PrNet wrapper for ablation experiments."""

    def __init__(self, debiasing_layer, **kwargs):
        super().__init__(**kwargs)
        self.debiasing_layer = debiasing_layer

    def forward(self, feature_inputs, *args):
        return self.debiasing_layer(feature_inputs, *args)


# Feature group definitions
# Based on DataPreprocessing_PrNet_parallel.py, the 16 features are:
# 0: CN0/50
# 1: sinE
# 2: cosE
# 3: PRN/32
# 4-6: not used in model (satellite positions — columns 4,5,6 in post_x)
# 7-9: WLS lon (3 values)
# 10-12: WLS lat (3 values)
# 13-15: Unit geometry vector (3 values)
# 16: smoothed PR residuals (label, not input)
# 17-19: Heading (3 values)

ABLATION_CONFIGS = {
    'no_cn0': {
        'name': 'Without CN0',
        'description': 'Remove CN0 (carrier-to-noise ratio)',
        'feature_indices': [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19],
        'input_size': 15,
    },
    'no_elevation': {
        'name': 'Without sinE/cosE',
        'description': 'Remove elevation angle features (sinE, cosE)',
        'feature_indices': [0, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19],
        'input_size': 14,
    },
    'no_prn': {
        'name': 'Without PRN',
        'description': 'Remove satellite PRN number',
        'feature_indices': [0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19],
        'input_size': 15,
    },
    'no_wls_pos': {
        'name': 'Without WLS Position',
        'description': 'Remove WLS-based position estimates (6 features: lon×3 + lat×3)',
        'feature_indices': [0, 1, 2, 3, 13, 14, 15, 17, 18, 19],
        'input_size': 10,
    },
    'no_geom_vec': {
        'name': 'Without Geometry Vector',
        'description': 'Remove unit geometry vector (N, E, D)',
        'feature_indices': [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 17, 18, 19],
        'input_size': 13,
    },
    'no_heading': {
        'name': 'Without Heading',
        'description': 'Remove smartphone heading (N, E, D)',
        'feature_indices': [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'input_size': 13,
    },
}


def ablated_data_preprocessing(enc_x, device, ablation_type):
    """
    Data preprocessing with feature ablation.
    Returns features with specified group removed.
    """
    # Same valid PRN index as original
    valid_prn_index = enc_x[:, :, 1] != 0

    post_enc_x_seq = []

    # Build all 22 feature channels as in original
    # 0: CN0/50
    cn0 = enc_x[:, :, 8].unsqueeze(-1) / 50
    # 1: sinE
    sinE = torch.sin(enc_x[:, :, 6]).unsqueeze(-1)
    # 2: cosE
    cosE = torch.cos(enc_x[:, :, 6]).unsqueeze(-1)
    # 3: PRN/32
    prn = enc_x[:, :, 1:2] / 32
    # 4-6: satellite positions (used in post_x but not in model forward)
    sv_pos = enc_x[:, :, 2:5]
    # 7-9: WLS lon
    wls_lon = enc_x[:, :, 22:25] / torch.tensor([180, 60, 60]).to(device)
    # 10-12: WLS lat
    wls_lat = enc_x[:, :, 25:28] / torch.tensor([90, 60, 60]).to(device)
    # 13-15: geometry vector
    geom = enc_x[:, :, 28:31]
    # 16: smoothed PR residuals (label)
    smooth_res = enc_x[:, :, 34:35]
    # 17-19: heading
    heading = enc_x[:, :, 35:38]
    # 20: last row of H matrix
    h_item = enc_x[:, :, 38:39]
    # 21: unsmoothed PR residuals
    unsmooth_res = enc_x[:, :, 31:32]

    # Map index to feature tensor
    feature_map = {
        0: cn0,
        1: sinE,
        2: cosE,
        3: prn,
        4: sv_pos[:, :, 0:1],
        5: sv_pos[:, :, 1:2],
        6: sv_pos[:, :, 2:3],
        7: wls_lon[:, :, 0:1],
        8: wls_lon[:, :, 1:2],
        9: wls_lon[:, :, 2:3],
        10: wls_lat[:, :, 0:1],
        11: wls_lat[:, :, 1:2],
        12: wls_lat[:, :, 2:3],
        13: geom[:, :, 0:1],
        14: geom[:, :, 1:2],
        15: geom[:, :, 2:3],
        16: smooth_res,
        17: heading[:, :, 0:1],
        18: heading[:, :, 1:2],
        19: heading[:, :, 2:3],
        20: h_item,
        21: unsmooth_res,
    }

    config = ABLATION_CONFIGS[ablation_type]
    # Build ablated feature set (only model input features, not label/aux)
    ablated_features = []
    for idx in config['feature_indices']:
        ablated_features.append(feature_map[idx])

    # Also append the non-ablated auxiliary features (16, 20, 21)
    # These are needed for loss computation and output
    ablated_features.append(smooth_res)   # will be at index len(config['feature_indices'])
    ablated_features.append(h_item)       # +1
    ablated_features.append(unsmooth_res)  # +2

    post_enc_x = torch.cat(ablated_features, dim=-1)
    return post_enc_x, valid_prn_index, config['input_size']


def train_ablated_model(ablation_type, data_iter, num_epochs=100, lr=0.01,
                        num_hiddens=40, num_layers=20, device=torch.device('cpu')):
    """Train an ablated PrNet model."""
    torch.set_default_tensor_type(torch.DoubleTensor)

    config = ABLATION_CONFIGS[ablation_type]
    input_size = config['input_size']

    extractor = AblatedMlpFeatureExtractor(
        input_size, num_hiddens, num_layers, dropout=0,
        ablation_type=ablation_type
    )
    model = AblatedPrNet(extractor).double()
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Ablation '{config['name']}': input_size={input_size}, params={total_params}")

    start = time.time()
    final_loss = 0

    for epoch in range(num_epochs):
        for batch in data_iter:
            optimizer.zero_grad()
            x, y = [z.to(device) for z in batch[:2]]
            post_x, valid_idx, _ = ablated_data_preprocessing(x, device, ablation_type)

            # Split features and auxiliary channels
            model_input = post_x[:, :, :input_size]
            # Auxiliary channels are at the end
            aux_start = input_size
            smooth_res = post_x[:, :, aux_start:aux_start + 1]
            h_item = post_x[:, :, aux_start + 1:aux_start + 2]

            preds = model(model_input)

            broadcast_idx = valid_idx.unsqueeze(-1)
            loss = loss_fn(
                (preds - torch.bmm(h_item.permute(0, 2, 1), preds))[broadcast_idx],
                smooth_res[broadcast_idx]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            final_loss = loss.item()

    train_time = time.time() - start
    print(f"    Training done in {train_time:.1f}s, final loss: {final_loss:.6f}")

    return model, total_params, train_time


def evaluate_ablated_model(model, ablation_type, data_iter, device=torch.device('cpu')):
    """Evaluate an ablated PrNet model."""
    config = ABLATION_CONFIGS[ablation_type]
    input_size = config['input_size']

    model.eval()
    output_seq = []
    loss_values = []
    inference_times = []
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        for batch in data_iter:
            x, y = [z.to(device) for z in batch]
            post_x, valid_idx, _ = ablated_data_preprocessing(x, device, ablation_type)

            model_input = post_x[:, :, :input_size]
            aux_start = input_size
            smooth_res = post_x[:, :, aux_start:aux_start + 1]
            h_item = post_x[:, :, aux_start + 1:aux_start + 2]

            start = time.time()
            preds = model(model_input)
            inference_times.append(time.time() - start)

            broadcast_idx = valid_idx.unsqueeze(-1)
            loss = loss_fn(
                (preds - torch.bmm(h_item.permute(0, 2, 1), preds))[broadcast_idx],
                smooth_res[broadcast_idx]
            )
            loss_values.append(loss.item())

            prmResi = preds - torch.bmm(h_item.permute(0, 2, 1), preds)
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
        return np.vstack(output_seq), loss_values, inference_times
    return np.array([]).reshape(0, 7), loss_values, inference_times


def run_all_ablation_experiments(train_data_iter_fn, eval_data_iter_fn,
                                 num_epochs=100, lr=0.01,
                                 device=torch.device('cpu')):
    """
    Run all 6 ablation experiments.
    
    train_data_iter_fn: callable that returns a fresh training data iterator
    eval_data_iter_fn: callable that returns a fresh evaluation data iterator
    
    Returns dict of ablation_type -> {model, params, train_time, predictions, losses, inference_times}
    """
    results = {}

    for abl_type, config in ABLATION_CONFIGS.items():
        print(f"\n  === Ablation: {config['name']} ===")
        print(f"  Description: {config['description']}")

        # Train
        model, params, train_time = train_ablated_model(
            abl_type, train_data_iter_fn(), num_epochs=num_epochs, lr=lr, device=device
        )

        # Evaluate
        predictions, losses, inference_times = evaluate_ablated_model(
            model, abl_type, eval_data_iter_fn(), device=device
        )

        avg_inference = np.mean(inference_times) if inference_times else 0
        avg_loss = np.mean(losses) if losses else 0

        results[abl_type] = {
            'name': config['name'],
            'description': config['description'],
            'input_size': config['input_size'],
            'num_params': params,
            'train_time': train_time,
            'predictions': predictions,
            'avg_loss': avg_loss,
            'avg_inference_time': avg_inference,
            'losses': losses,
        }

        print(f"  Results: avg_loss={avg_loss:.6f}, avg_inference={avg_inference:.6f}s")

    return results
