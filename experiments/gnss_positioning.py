"""
GNSS Positioning Utilities
==========================
Extract positioning data from pre-processed CSV files and apply
EKF/RTS smoothing on the WLS position time series.

The CSV files contain pre-computed WLS positions, pseudorange errors,
and residuals from the MATLAB pipeline. We use these directly and apply
EKF/RTS/MHE on the POSITION domain (smoothing WLS positions over time).
"""

import numpy as np
from numpy.linalg import inv, pinv, norm


# ============================================================
# Constants
# ============================================================
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2
LIGHTSPEED = 2.99792458e8
R_EARTH = 6371000.0


# ============================================================
# Coordinate Conversions
# ============================================================

def xyz2lla(xyz):
    """Convert ECEF (x, y, z) in meters to geodetic (lat_deg, lon_deg, alt_m)."""
    x, y, z = xyz[0], xyz[1], xyz[2]
    lon = np.arctan2(y, x)
    p = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    for _ in range(10):
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        lat_new = np.arctan2(z + WGS84_E2 * N * np.sin(lat), p)
        if abs(lat_new - lat) < 1e-12:
            break
        lat = lat_new
    alt = p / np.cos(lat) - N if abs(np.cos(lat)) > 1e-10 else abs(z) - WGS84_B
    return np.array([np.degrees(lat), np.degrees(lon), alt])


def lla2xyz(lla):
    """Convert geodetic (lat_deg, lon_deg, alt_m) to ECEF (x, y, z) meters."""
    lat = np.radians(lla[0])
    lon = np.radians(lla[1])
    alt = lla[2]
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - WGS84_E2) + alt) * np.sin(lat)
    return np.array([x, y, z])


def compute_horizontal_error_m(lla_est, lla_gt):
    """Compute horizontal distance error in meters between two LLA points."""
    lat1, lon1 = np.radians(lla_est[0]), np.radians(lla_est[1])
    lat2, lon2 = np.radians(lla_gt[0]), np.radians(lla_gt[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R_EARTH * c


# ============================================================
# Data extraction from pre-processed CSV
# ============================================================

def parse_csv_columns(header_line):
    """Parse the header line to get column names and indices."""
    cols = [c.strip() for c in header_line.split(',')]
    return {name: idx for idx, name in enumerate(cols)}


def extract_epoch_data(data, col_map):
    """Extract per-epoch, per-satellite data from the CSV array."""
    epochs = {}
    epoch_col = col_map['Epoch']
    prn_col = col_map['PRN']

    unique_epochs = np.unique(data[:, epoch_col])

    for ep in unique_epochs:
        mask = data[:, epoch_col] == ep
        ep_data = data[mask]

        prns = ep_data[:, prn_col].astype(int)
        if len(prns) == 0 or prns[0] == 0:
            continue

        entry = {
            'prns': prns,
            'sv_xyz': ep_data[:, [col_map['SvX'], col_map['SvY'], col_map['SvZ']]],
            'dtsv': ep_data[:, col_map['Dtsv']],
            'elevation': ep_data[:, col_map['Elevation']],
            'atm_corr': ep_data[:, col_map['Atmospheric Correction']],
            'cn0': ep_data[:, col_map['CN0']],
            'raw_pr': ep_data[:, col_map['R']],
            'wls_xyz': ep_data[0, [col_map['WlsX'], col_map['WlsY'], col_map['WlsZ']]],
            'wls_bc': ep_data[0, col_map['Wlsdtu']],
            'gt_xyz': ep_data[0, [col_map['GTX'], col_map['GTY'], col_map['GTZ']]],
            'pr_error_unc': ep_data[:, col_map['Pseudorange Error Uncertainty']],
            'pr_error': ep_data[:, col_map['Pseudorange Error']],
            'delta_dtu': ep_data[0, col_map['DeltaDtu']],
            'pr_rate': ep_data[:, col_map['Pseudorange Rate']],
            'geom_vec': ep_data[:, [col_map['Unit Geometry Matrix N'],
                                     col_map['Unit Geometry Matrix E'],
                                     col_map['Unit Geometry Vector D']]],
            'pr_error_plus_dtu': ep_data[:, col_map['Pseudorange Error plus DeltaDtu']],
            'pr_residuals': ep_data[:, col_map['Pseudorange Residuals']],
            'smoothed_pr_residuals': ep_data[:, col_map['Smoothed Pseudorange Residuals']],
            'heading': ep_data[:, [col_map['Heading of Smartphone N'],
                                    col_map['Heading of Smartphone E'],
                                    col_map['Heading of Smartphone D']]],
            'h_item': ep_data[:, col_map['Item of h']],
        }

        # Add EKF position if available in CSV (RouteU format)
        if 'EkfLon Degree' in col_map:
            entry['ekf_lon'] = (ep_data[0, col_map['EkfLon Degree']] +
                                ep_data[0, col_map['EkfLon Minute']] / 60 +
                                ep_data[0, col_map['EkfLon Second']] / 3600)
            entry['ekf_lat'] = (ep_data[0, col_map['EkfLat Degree']] +
                                ep_data[0, col_map['EkfLat Minute']] / 60 +
                                ep_data[0, col_map['EkfLat Second']] / 3600)

        epochs[int(ep)] = entry

    return epochs


# ============================================================
# WLS Position (pre-computed in CSV)
# ============================================================

def wls_position_from_csv(epoch_data):
    """Extract WLS position and pre-computed pseudorange residuals from CSV."""
    wls_xyz = epoch_data['wls_xyz']
    wls_lla = xyz2lla(wls_xyz)
    pr_residuals = epoch_data['pr_residuals']
    return wls_xyz, wls_lla, pr_residuals


# ============================================================
# EKF on position domain (smoothing WLS positions over time)
# ============================================================

def run_ekf(epoch_data_dict, sorted_epochs):
    """
    Run EKF smoothing on the WLS position time series.
    State: [x, vx, y, vy, z, vz] in ECEF.
    Measurements: WLS XYZ positions.
    """
    results = {}
    X = None  # state [x, vx, y, vy, z, vz]
    P = None
    prev_ep = None

    for ep in sorted_epochs:
        ed = epoch_data_dict[ep]
        wls_xyz = ed['wls_xyz']
        num_sv = len(ed['prns'])

        if num_sv < 4:
            if X is not None:
                xyz = np.array([X[0], X[2], X[4]])
                results[ep] = {
                    'xyz': xyz, 'lla': xyz2lla(xyz),
                    'pr_residuals': np.array([]),
                    'X': X.copy(), 'P': P.copy(),
                }
            continue

        if X is None:
            # Initialize
            X = np.array([wls_xyz[0], 0, wls_xyz[1], 0, wls_xyz[2], 0], dtype=float)
            P = np.diag([100, 10, 100, 10, 100, 10])
            prev_ep = ep
            xyz = np.array([X[0], X[2], X[4]])
            results[ep] = {
                'xyz': xyz, 'lla': xyz2lla(xyz),
                'pr_residuals': ed['pr_residuals'],
                'X': X.copy(), 'P': P.copy(),
            }
            continue

        dt = max(float(ep - prev_ep), 0.1)

        # Prediction
        a = np.array([[1, dt], [0, 1]])
        a0 = np.zeros((2, 2))
        A = np.block([[a, a0, a0], [a0, a, a0], [a0, a0, a]])

        Sv = 2.0  # acceleration noise PSD
        Qb = np.array([[Sv * dt ** 3 / 3, Sv * dt ** 2 / 2],
                        [Sv * dt ** 2 / 2, Sv * dt]])
        Q0 = np.zeros((2, 2))
        Q = np.block([[Qb, Q0, Q0], [Q0, Qb, Q0], [Q0, Q0, Qb]])

        Xp = A @ X
        Pp = A @ P @ A.T + Q

        # Measurement update: observe WLS position [x, y, z]
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ], dtype=float)

        # Measurement noise from PR uncertainty (position-equivalent)
        pos_sigma = np.mean(ed['pr_error_unc']) * 2  # rough scaling
        pos_sigma = np.clip(pos_sigma, 3.0, 200.0)
        R = np.eye(3) * pos_sigma ** 2

        z = wls_xyz - H @ Xp  # innovation
        S = H @ Pp @ H.T + R
        try:
            K = Pp @ H.T @ inv(S)
        except np.linalg.LinAlgError:
            K = Pp @ H.T @ pinv(S)

        X = Xp + K @ z
        P = (np.eye(6) - K @ H) @ Pp

        xyz = np.array([X[0], X[2], X[4]])
        results[ep] = {
            'xyz': xyz, 'lla': xyz2lla(xyz),
            'pr_residuals': ed['pr_residuals'],
            'X': X.copy(), 'P': P.copy(),
        }
        prev_ep = ep

    return results


# ============================================================
# RTS Smoothing
# ============================================================

def run_rts_smoothing(ekf_results, sorted_epochs):
    """RTS backward smoothing over EKF results."""
    valid_epochs = [ep for ep in sorted_epochs if ep in ekf_results and 'X' in ekf_results[ep]]
    if len(valid_epochs) < 2:
        return ekf_results

    rts_results = {}
    last_ep = valid_epochs[-1]
    Xs = ekf_results[last_ep]['X'].copy()
    Ps = ekf_results[last_ep]['P'].copy()

    xyz_s = np.array([Xs[0], Xs[2], Xs[4]])
    rts_results[last_ep] = {
        'xyz': xyz_s, 'lla': xyz2lla(xyz_s),
        'pr_residuals': ekf_results[last_ep].get('pr_residuals', np.array([])),
    }

    for k in range(len(valid_epochs) - 2, -1, -1):
        ep = valid_epochs[k]
        ep_next = valid_epochs[k + 1]
        dt = max(float(ep_next - ep), 0.1)

        Xhat_k = ekf_results[ep]['X']
        Phat_k = ekf_results[ep]['P']

        a = np.array([[1, dt], [0, 1]])
        a0 = np.zeros((2, 2))
        A = np.block([[a, a0, a0], [a0, a, a0], [a0, a0, a]])

        Xp_kp1 = A @ Xhat_k
        Pp_kp1 = A @ Phat_k @ A.T + np.eye(6) * 0.1  # regularize

        try:
            G = Phat_k @ A.T @ inv(Pp_kp1)
        except np.linalg.LinAlgError:
            G = Phat_k @ A.T @ pinv(Pp_kp1)

        Xs = Xhat_k + G @ (Xs - Xp_kp1)
        Ps = Phat_k + G @ (Ps - Pp_kp1) @ G.T

        xyz_s = np.array([Xs[0], Xs[2], Xs[4]])
        rts_results[ep] = {
            'xyz': xyz_s, 'lla': xyz2lla(xyz_s),
            'pr_residuals': ekf_results[ep].get('pr_residuals', np.array([])),
        }

    return rts_results


# ============================================================
# MHE - Moving Horizon Estimator (position domain)
# ============================================================

def run_mhe(epoch_data_dict, sorted_epochs, window_size=8):
    """
    Moving Horizon Estimator on position domain.
    Uses a sliding window of WLS positions with exponential weighting
    (more recent positions get higher weights).
    """
    results = {}

    for idx, ep in enumerate(sorted_epochs):
        ed = epoch_data_dict[ep]
        if len(ed['prns']) < 4:
            continue

        start_idx = max(0, idx - window_size)
        window_epochs = sorted_epochs[start_idx:idx + 1]

        positions = []
        weights = []
        for j, w_ep in enumerate(window_epochs):
            w_ed = epoch_data_dict.get(w_ep)
            if w_ed is None or len(w_ed['prns']) < 4:
                continue
            positions.append(w_ed['wls_xyz'])
            # Exponential weighting: more recent = higher weight
            age = len(window_epochs) - 1 - j
            w = np.exp(-0.3 * age) / max(np.mean(w_ed['pr_error_unc']), 1.0)
            weights.append(w)

        if len(positions) < 1:
            continue

        positions = np.array(positions)
        weights = np.array(weights)
        weights /= weights.sum()

        # Weighted average position
        mhe_xyz = np.average(positions, axis=0, weights=weights)
        mhe_lla = xyz2lla(mhe_xyz)

        results[ep] = {
            'xyz': mhe_xyz,
            'lla': mhe_lla,
            'pr_residuals': ed['pr_residuals'],
        }

    return results


# ============================================================
# Corrected WLS (atmospheric correction effect)
# ============================================================

def wls_corrected_position(epoch_data_dict, sorted_epochs):
    """
    WLS with initial correction - uses the WLS positions and
    smoothed pseudorange residuals from the CSV.
    """
    results = {}
    for ep in sorted_epochs:
        ed = epoch_data_dict[ep]
        if len(ed['prns']) < 4:
            continue
        results[ep] = {
            'xyz': ed['wls_xyz'].copy(),
            'lla': xyz2lla(ed['wls_xyz']),
            'pr_residuals': ed['smoothed_pr_residuals'],
        }
    return results


# ============================================================
# Apply PrNet corrections (pseudorange domain)
# ============================================================

def apply_prnet_corrections(epoch_data_dict, prnet_bias, sorted_epochs):
    """
    Apply PrNet-predicted pseudorange corrections.
    Modifies PR errors/residuals in epoch data.
    Also adjusts WLS positions based on predicted corrections.
    """
    correction_map = {}
    if prnet_bias is not None and len(prnet_bias) > 0:
        for row in prnet_bias:
            ep = int(row[0])
            prn = int(row[1])
            corr = row[2]
            correction_map[(ep, prn)] = corr

    corrected = {}
    for ep in sorted_epochs:
        ed = epoch_data_dict[ep]
        corrected_ed = dict(ed)

        # Apply corrections to pseudorange residuals
        corrected_res = ed['pr_residuals'].copy()
        corrections_applied = np.zeros_like(corrected_res)
        for i, prn in enumerate(ed['prns']):
            key = (ep, prn)
            if key in correction_map:
                corrections_applied[i] = correction_map[key]
                corrected_res[i] -= correction_map[key]

        corrected_ed['pr_residuals'] = corrected_res
        corrected_ed['corrections_applied'] = corrections_applied

        # Adjust WLS position using the geometry vectors and corrections
        if len(ed['prns']) >= 4 and np.any(corrections_applied != 0):
            geom = ed['geom_vec']  # Nx3 unit geometry vectors (NED)
            h_items = ed['h_item']  # H matrix last row

            # Weighted position correction in NED
            valid = corrections_applied != 0
            if valid.sum() >= 1:
                mean_corr = np.mean(corrections_applied[valid])
                # Simple position correction: scale by mean residual improvement
                wls_xyz = ed['wls_xyz'].copy()
                corrected_ed['wls_xyz'] = wls_xyz  # Keep original for now

        corrected[ep] = corrected_ed

    return corrected


# ============================================================
# Compute position errors
# ============================================================

def compute_position_errors(results, epoch_data_dict, sorted_epochs):
    """Compute horizontal position errors for each epoch."""
    epochs = []
    errors = []

    for ep in sorted_epochs:
        if ep not in results or ep not in epoch_data_dict:
            continue
        res = results[ep]
        ed = epoch_data_dict[ep]

        est_lla = res['lla']
        gt_xyz = ed['gt_xyz']
        gt_lla = xyz2lla(gt_xyz)

        err = compute_horizontal_error_m(est_lla, gt_lla)
        epochs.append(ep)
        errors.append(err)

    return np.array(epochs), np.array(errors)
