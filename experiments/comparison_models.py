"""
Comparison Models for Pseudorange Correction
=============================================
Implements PBC-RF, FCNN-LSTM, and Set Transformer for comparison with PrNet.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# PBC-RF: Point-Based Correction using Random Forest
# ============================================================

class PBCRF:
    """
    Point-Based Correction Random Forest.
    Uses scikit-learn RandomForestRegressor for per-satellite pseudorange correction.
    Same input features as PrNet (16 features per satellite).
    """

    def __init__(self, n_estimators=100, max_depth=20, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _extract_features_labels(self, data_iter, device):
        """Extract flat feature/label arrays from PrNet-style data iterator."""
        from DataPreprocessing_PrNet_parallel import data_preprocessing

        all_features = []
        all_labels = []

        for batch in data_iter:
            x, y = [z.to(device) for z in batch]
            post_x, valid_idx = data_preprocessing(x, device)

            for i in range(x.shape[0]):
                mask = valid_idx[i]
                if mask.sum() == 0:
                    continue
                # 16 input features (same as PrNet): indices 0-3, 7-15, 17-19
                feats = torch.cat([
                    post_x[i, mask, :4],
                    post_x[i, mask, 7:16],
                    post_x[i, mask, 17:20]
                ], dim=-1).cpu().numpy()

                labels = post_x[i, mask, 16:17].cpu().numpy().ravel()
                all_features.append(feats)
                all_labels.append(labels)

        if len(all_features) == 0:
            return np.array([]).reshape(0, 16), np.array([])

        return np.vstack(all_features), np.concatenate(all_labels)

    def train(self, data_iter, device=torch.device('cpu')):
        """Train the RF model."""
        X, y = self._extract_features_labels(data_iter, device)
        if len(X) == 0:
            return
        start = time.time()
        self.model.fit(X, y)
        self.train_time = time.time() - start
        self.is_fitted = True

    def predict_from_iter(self, data_iter, device=torch.device('cpu')):
        """Predict pseudorange corrections using the same data iterator format."""
        from DataPreprocessing_PrNet_parallel import data_preprocessing

        output_seq = []
        inference_times = []

        for batch in data_iter:
            x, y = [z.to(device) for z in batch]
            post_x, valid_idx = data_preprocessing(x, device)

            for i in range(x.shape[0]):
                mask = valid_idx[i]
                if mask.sum() == 0:
                    continue

                feats = torch.cat([
                    post_x[i, mask, :4],
                    post_x[i, mask, 7:16],
                    post_x[i, mask, 17:20]
                ], dim=-1).cpu().numpy()

                start = time.time()
                preds = self.model.predict(feats)
                inference_times.append(time.time() - start)

                # Build output: [epoch, prn, prediction, smoothed_residual, unsmoothed_residual]
                enc_x = x[i]
                epochs = enc_x[mask, 0].cpu().numpy()
                prns = enc_x[mask, 1].cpu().numpy()
                smoothed_res = enc_x[mask, 34].cpu().numpy()  # smoothed PR residuals
                unsmoothed_res = enc_x[mask, 31].cpu().numpy()

                out = np.column_stack([epochs, prns, preds, smoothed_res, unsmoothed_res])
                output_seq.append(out)

        if output_seq:
            return np.vstack(output_seq), inference_times
        return np.array([]).reshape(0, 5), inference_times

    @property
    def num_parameters(self):
        """Approximate parameter count for RF (number of nodes across trees)."""
        if not self.is_fitted:
            return 0
        total = 0
        for tree in self.model.estimators_:
            total += tree.tree_.node_count
        return total


# ============================================================
# FCNN-LSTM: Fully Connected NN + LSTM
# ============================================================

class FCNNLSTM(nn.Module):
    """
    FCNN-LSTM model for pseudorange correction.
    Architecture: FC layers -> LSTM -> FC output
    Processes satellite features with FC, then uses LSTM for temporal modeling.
    """

    def __init__(self, input_size=16, fc_hidden=64, lstm_hidden=64,
                 num_fc_layers=3, num_lstm_layers=2, dropout=0.1):
        super().__init__()

        # FC feature extractor (per-satellite)
        fc_layers = []
        in_dim = input_size
        for _ in range(num_fc_layers):
            fc_layers.extend([
                nn.Linear(in_dim, fc_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = fc_hidden
        self.fc_encoder = nn.Sequential(*fc_layers)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=fc_hidden,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # Output layer
        self.fc_output = nn.Linear(lstm_hidden, 1)

        self.fc_hidden = fc_hidden
        self.lstm_hidden = lstm_hidden

    def forward(self, x):
        """
        x: (batch_size, prn_size, input_size)
        Returns: (batch_size, prn_size, 1)
        """
        batch_size, prn_size, _ = x.shape

        # Apply FC to each satellite independently
        # Reshape: (batch*prn, input_size)
        x_flat = x.reshape(-1, x.shape[-1])
        fc_out = self.fc_encoder(x_flat)
        # Reshape back: (batch, prn, fc_hidden)
        fc_out = fc_out.reshape(batch_size, prn_size, -1)

        # LSTM: treat satellites as sequence
        lstm_out, _ = self.lstm(fc_out)

        # Output
        output = self.fc_output(lstm_out)  # (batch, prn, 1)
        return output


class FCNNLSTMWrapper:
    """Wrapper for training and evaluation of FCNN-LSTM."""

    def __init__(self, input_size=16, **kwargs):
        torch.set_default_tensor_type(torch.DoubleTensor)
        self.model = FCNNLSTM(input_size=input_size, **kwargs).double()
        self.device = torch.device('cpu')

    def train(self, data_iter, num_epochs=100, lr=0.01):
        from DataPreprocessing_PrNet_parallel import data_preprocessing

        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        start = time.time()
        for epoch in range(num_epochs):
            for batch in data_iter:
                optimizer.zero_grad()
                x, y = [z.to(self.device) for z in batch[:2]]
                post_x, valid_idx = data_preprocessing(x, self.device)

                # Extract 16 features
                features = torch.cat([
                    post_x[:, :, :4],
                    post_x[:, :, 7:16],
                    post_x[:, :, 17:20]
                ], dim=-1)

                preds = self.model(features)

                broadcast_idx = valid_idx.unsqueeze(-1)
                loss = loss_fn(
                    (preds - torch.bmm(post_x[:, :, 20:21].permute(0, 2, 1), preds))[broadcast_idx],
                    post_x[:, :, 16:17][broadcast_idx]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

        self.train_time = time.time() - start

    def evaluate(self, data_iter):
        from DataPreprocessing_PrNet_parallel import data_preprocessing

        self.model.eval()
        output_seq = []
        inference_times = []

        with torch.no_grad():
            for batch in data_iter:
                x, y = [z.to(self.device) for z in batch]
                post_x, valid_idx = data_preprocessing(x, self.device)

                features = torch.cat([
                    post_x[:, :, :4],
                    post_x[:, :, 7:16],
                    post_x[:, :, 17:20]
                ], dim=-1)

                start = time.time()
                preds = self.model(features)
                inference_times.append(time.time() - start)

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
            return np.vstack(output_seq), inference_times
        return np.array([]).reshape(0, 7), inference_times

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ============================================================
# Set Transformer
# ============================================================

class MAB(nn.Module):
    """Multihead Attention Block."""

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.ln0 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(dim_V) if ln else nn.Identity()

    def forward(self, Q, K):
        Q_ = self.fc_q(Q)
        K_ = self.fc_k(K)
        V_ = self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q_.split(dim_split, -1), 0)
        K_ = torch.cat(K_.split(dim_split, -1), 0)
        V_ = torch.cat(V_.split(dim_split, -1), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / np.sqrt(dim_split), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = self.ln0(O)
        O = O + nn.functional.relu(self.fc_o(O))
        O = self.ln1(O)
        return O


class SAB(nn.Module):
    """Set Attention Block."""

    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class PMA(nn.Module):
    """Pooling by Multihead Attention."""

    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim).double())
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformerModel(nn.Module):
    """
    Set Transformer for pseudorange correction.
    Uses SAB layers for set encoding and PMA for aggregation,
    then broadcasts back to per-element predictions.
    """

    def __init__(self, input_size=16, dim_hidden=64, num_heads=4,
                 num_sab_layers=2, num_seeds=1, ln=True):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_size, dim_hidden)

        # SAB layers
        sab_layers = []
        for _ in range(num_sab_layers):
            sab_layers.append(SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        self.encoder = nn.Sequential(*sab_layers)

        # PMA for set-level aggregation
        self.pma = PMA(dim_hidden, num_heads, num_seeds, ln=ln)

        # Decoder: broadcast aggregated representation + per-element features
        self.decoder = nn.Sequential(
            nn.Linear(dim_hidden * 2, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1)
        )

        self.dim_hidden = dim_hidden

    def forward(self, x):
        """
        x: (batch_size, prn_size, input_size)
        Returns: (batch_size, prn_size, 1)
        """
        batch_size, prn_size, _ = x.shape

        # Project input
        h = self.input_proj(x)  # (B, N, D)

        # Encode with SAB
        h_enc = self.encoder(h)  # (B, N, D)

        # Aggregate with PMA
        h_agg = self.pma(h_enc)  # (B, num_seeds, D)

        # Broadcast aggregated representation
        h_agg_broadcast = h_agg.expand(-1, prn_size, -1)  # (B, N, D)

        # Concatenate per-element features with aggregated
        h_combined = torch.cat([h_enc, h_agg_broadcast], dim=-1)  # (B, N, 2D)

        # Decode to per-element prediction
        output = self.decoder(h_combined)  # (B, N, 1)
        return output


class SetTransformerWrapper:
    """Wrapper for training and evaluation of Set Transformer."""

    def __init__(self, input_size=16, **kwargs):
        torch.set_default_tensor_type(torch.DoubleTensor)
        self.model = SetTransformerModel(input_size=input_size, **kwargs).double()
        self.device = torch.device('cpu')

    def train(self, data_iter, num_epochs=100, lr=0.01):
        from DataPreprocessing_PrNet_parallel import data_preprocessing

        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        start = time.time()
        for epoch in range(num_epochs):
            for batch in data_iter:
                optimizer.zero_grad()
                x, y = [z.to(self.device) for z in batch[:2]]
                post_x, valid_idx = data_preprocessing(x, self.device)

                features = torch.cat([
                    post_x[:, :, :4],
                    post_x[:, :, 7:16],
                    post_x[:, :, 17:20]
                ], dim=-1)

                preds = self.model(features)

                broadcast_idx = valid_idx.unsqueeze(-1)
                loss = loss_fn(
                    (preds - torch.bmm(post_x[:, :, 20:21].permute(0, 2, 1), preds))[broadcast_idx],
                    post_x[:, :, 16:17][broadcast_idx]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

        self.train_time = time.time() - start

    def evaluate(self, data_iter):
        from DataPreprocessing_PrNet_parallel import data_preprocessing

        self.model.eval()
        output_seq = []
        inference_times = []

        with torch.no_grad():
            for batch in data_iter:
                x, y = [z.to(self.device) for z in batch]
                post_x, valid_idx = data_preprocessing(x, self.device)

                features = torch.cat([
                    post_x[:, :, :4],
                    post_x[:, :, 7:16],
                    post_x[:, :, 17:20]
                ], dim=-1)

                start = time.time()
                preds = self.model(features)
                inference_times.append(time.time() - start)

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
            return np.vstack(output_seq), inference_times
        return np.array([]).reshape(0, 7), inference_times

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
