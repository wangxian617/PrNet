"""
GNSS Positioning Utilities
==========================
Python implementation of WLS, EKF, RTS, and MHE positioning algorithms,
ported from the MATLAB codebase in GNSS_opensource_software/.

These operate on the pre-processed CSV data already available in
Data/RouteR/Testing/ and Data/RouteU/Testing/.
"""

import numpy as np
from numpy.linalg import inv, pinv, norm


# ============================================================
# Coordinate Conversions (from Xyz2Lla.m, Lla2Xyz.m, RotEcef2Ned.m)
# ============================================================

# WGS-84 constants
WGS84_A = 6378137.0  # semi-major axis (m)
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2


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


def rot_ecef2ned(lat_deg, lon_deg):
    """Rotation matrix from ECEF to NED, given lat/lon in degrees."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    R = np.array([
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [-np.sin(lon), np.cos(lon), 0],
        [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]
    ])
    return R


def compute_horizontal_error_m(lla_est, lla_gt):
    """Compute horizontal distance error in meters between two LLA points."""
    lat1, lon1 = np.radians(lla_est[0]), np.radians(lla_est[1])
    lat2, lon2 = np.radians(lla_gt[0]), np.radians(lla_gt[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371000.0 * c


# ============================================================
# Data extraction from pre-processed CSV
# ============================================================

def parse_csv_columns(header_line):
    """Parse the header line to get column names and indices."""
    cols = [c.strip() for c in header_line.split(',')]
    return {name: idx for idx, name in enumerate(cols)}


def extract_epoch_data(data, col_map):
    """
    Extract per-epoch, per-satellite data from the CSV array.
    
    Returns dict of epoch -> {
        'prns': array of PRN,
        'sv_xyz': Nx3 satellite positions,
        'elevation': N elevations,
        'cn0': N CN0 values,
        'raw_pr': N raw pseudoranges (R column),
        'atm_corr': N atmospheric corrections,
        'pr_error': N pseudorange errors (ground truth),
        'pr_residuals': N pseudorange residuals,
        'smoothed_pr_residuals': N smoothed pseudorange residuals,
        'wls_xyz': 3-vector WLS position (ECEF),
        'gt_xyz': 3-vector ground truth position (ECEF),
        'wls_bc': WLS clock bias,
        'heading': Nx3 heading vectors,
        'geom_vec': Nx3 unit geometry vectors,
        'pr_error_plus_dtu': N pseudorange error + delta dtu,
        'delta_dtu': scalar delta dtu,
    }
    """
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
            
        epochs[int(ep)] = {
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
            'hat_delta_dtu': ep_data[0, col_map['Hat_DeltaDtu']],
            'pr_rate': ep_data[:, col_map['Pseudorange Rate']],
            'geom_vec': ep_data[:, [col_map['Unit Geometry Matrix N'],
                                     col_map['Unit Geometry Matrix E'],
                                     col_map['Unit Geometry Vector D']]],
            'pr_error_plus_dtu': ep_data[:, col_map['Pseudorange Error plus DeltaDtu']],
            'azimuth': ep_data[:, col_map['Azimuth']],
            'pr_residuals': ep_data[:, col_map['Pseudorange Residuals']],
            'smoothed_pr_residuals': ep_data[:, col_map['Smoothed Pseudorange Residuals']],
            'heading': ep_data[:, [col_map['Heading of Smartphone N'],
                                    col_map['Heading of Smartphone E'],
                                    col_map['Heading of Smartphone D']]],
            'h_item': ep_data[:, col_map['Item of h']],
        }
    
    return epochs


# ============================================================
# WLS Position from pre-computed data
# ============================================================

def wls_position_from_csv(epoch_data):
    """
    Compute WLS position from pre-processed CSV data.
    Uses the WLS position already computed in the CSV (WlsX, WlsY, WlsZ).
    Also computes pseudorange residuals.
    
    Returns (wls_xyz, wls_lla, pr_residuals_wls)
    """
    wls_xyz = epoch_data['wls_xyz']
    wls_lla = xyz2lla(wls_xyz)
    
    # Compute pseudorange residuals from WLS solution
    sv_xyz = epoch_data['sv_xyz']
    raw_pr = epoch_data['raw_pr']
    wls_bc = epoch_data['wls_bc']
    
    # Geometric range from WLS position to each satellite
    ranges = np.sqrt(np.sum((sv_xyz - wls_xyz[np.newaxis, :]) ** 2, axis=1))
    
    # Pseudorange residuals = observed - predicted
    pr_residuals = raw_pr - (ranges + wls_bc)
    
    return wls_xyz, wls_lla, pr_residuals


# ============================================================
# EKF Positioning (simplified, using pre-computed data)
# ============================================================

class EKFState:
    """Extended Kalman Filter state for GNSS positioning."""
    
    def __init__(self):
        # State: [x, vx, y, vy, z, vz, bc, fc]
        self.X = np.zeros(8)
        self.P = np.eye(8) * 1e6
        self.initialized = False
        self.warmup_count = 0
    
    def initialize(self, wls_xyz, wls_bc):
        self.X = np.array([
            wls_xyz[0], 0.0,
            wls_xyz[1], 0.0,
            wls_xyz[2], 0.0,
            wls_bc, 0.0
        ])
        self.P = np.eye(8) * 100.0
        self.initialized = True
        self.warmup_count = 1
    
    def predict(self, dt):
        """State prediction step."""
        if dt <= 0:
            dt = 1.0
        
        # State transition matrix
        a = np.array([[1, dt], [0, 1]])
        a0 = np.zeros((2, 2))
        A = np.block([
            [a, a0, a0, a0],
            [a0, a, a0, a0],
            [a0, a0, a, a0],
            [a0, a0, a0, a]
        ])
        
        # Process noise - use moderate values
        Sv = 5.0  # velocity noise PSD (m/s^2)^2/Hz
        St = 100.0  # clock bias noise
        Sf = 10.0   # clock drift noise
        
        Qx = np.array([[Sv * dt ** 3 / 3, Sv * dt ** 2 / 2],
                        [Sv * dt ** 2 / 2, Sv * dt]])
        Qt = np.array([[St * dt + Sf * dt ** 3 / 3, Sf * dt ** 2 / 2],
                        [Sf * dt ** 2 / 2, Sf * dt]])
        Q0 = np.zeros((2, 2))
        Q = np.block([
            [Qx, Q0, Q0, Q0],
            [Q0, Qx, Q0, Q0],
            [Q0, Q0, Qx, Q0],
            [Q0, Q0, Q0, Qt]
        ])
        
        self.X = A @ self.X
        self.P = A @ self.P @ A.T + Q
        
        return self.X.copy(), self.P.copy()
    
    def update(self, sv_xyz, pr_observed, pr_sigma):
        """Measurement update step."""
        num_sv = len(pr_observed)
        if num_sv < 4:
            return self.X.copy()
        
        xyz = np.array([self.X[0], self.X[2], self.X[4]])
        bc = self.X[6]
        
        # Compute expected ranges and measurement matrix
        dxyz = xyz[np.newaxis, :] - sv_xyz  # user - satellite
        ranges = np.sqrt(np.sum(dxyz ** 2, axis=1))
        
        # Line of sight unit vectors
        los = dxyz / ranges[:, np.newaxis]
        
        # Measurement matrix C (2*numSv x 8) for position and velocity
        C = np.zeros((num_sv, 8))
        for i in range(num_sv):
            C[i, :] = [los[i, 0], 0, los[i, 1], 0, los[i, 2], 0, 1, 0]
        
        # Measurement residuals
        pr_predicted = ranges + bc
        z = pr_observed - pr_predicted
        
        # Measurement noise covariance
        R = np.diag(pr_sigma ** 2)
        
        # Kalman gain
        S = C @ self.P @ C.T + R
        try:
            K = self.P @ C.T @ inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ C.T @ pinv(S)
        
        # State update
        self.X = self.X + K @ z
        self.P = (np.eye(8) - K @ C) @ self.P
        self.warmup_count += 1
        
        return self.X.copy()
    
    def get_position_lla(self):
        xyz = np.array([self.X[0], self.X[2], self.X[4]])
        return xyz2lla(xyz)
    
    def get_position_xyz(self):
        return np.array([self.X[0], self.X[2], self.X[4]])


def run_ekf(epoch_data_dict, sorted_epochs):
    """
    Run EKF over all epochs.
    Returns dict of epoch -> {'xyz': ..., 'lla': ..., 'pr_residuals': ...}
    """
    ekf = EKFState()
    results = {}
    prev_epoch = None
    
    for ep in sorted_epochs:
        ed = epoch_data_dict[ep]
        num_sv = len(ed['prns'])
        
        if num_sv < 4:
            if ekf.initialized:
                results[ep] = {
                    'xyz': ekf.get_position_xyz(),
                    'lla': ekf.get_position_lla(),
                    'pr_residuals': np.array([]),
                    'bc': ekf.X[6],
                }
            continue
        
        if not ekf.initialized:
            ekf.initialize(ed['wls_xyz'], ed['wls_bc'])
            results[ep] = {
                'xyz': ekf.get_position_xyz(),
                'lla': ekf.get_position_lla(),
                'pr_residuals': ed['pr_residuals'],
                'bc': ekf.X[6],
            }
            prev_epoch = ep
            continue
        
        # Prediction
        dt = (ep - prev_epoch) if prev_epoch is not None else 1.0
        ekf.predict(dt)
        
        # Compute corrected pseudoranges
        raw_pr = ed['raw_pr']
        pr_sigma = ed['pr_error_unc']
        pr_sigma = np.clip(pr_sigma, 1.0, 100.0)
        
        # Update
        ekf.update(ed['sv_xyz'], raw_pr, pr_sigma)
        
        # Compute pseudorange residuals with EKF position
        ekf_xyz = ekf.get_position_xyz()
        ranges = np.sqrt(np.sum((ed['sv_xyz'] - ekf_xyz[np.newaxis, :]) ** 2, axis=1))
        pr_res = raw_pr - (ranges + ekf.X[6])
        
        results[ep] = {
            'xyz': ekf_xyz.copy(),
            'lla': ekf.get_position_lla(),
            'pr_residuals': pr_res,
            'bc': ekf.X[6],
            'X': ekf.X.copy(),
            'P': ekf.P.copy(),
            'Xp': ekf.X.copy(),  # store for RTS
            'Pp': ekf.P.copy(),
        }
        prev_epoch = ep
    
    return results


# ============================================================
# RTS Smoothing (from SmoothingKF.m)
# ============================================================

def run_rts_smoothing(ekf_results, sorted_epochs):
    """
    RTS backward smoothing over EKF results.
    Returns dict of epoch -> {'xyz': ..., 'lla': ...}
    """
    # Collect epochs that have valid EKF results with state
    valid_epochs = [ep for ep in sorted_epochs if ep in ekf_results and 'X' in ekf_results[ep]]
    
    if len(valid_epochs) < 2:
        return ekf_results
    
    rts_results = {}
    
    # Initialize with last epoch
    last_ep = valid_epochs[-1]
    Xs = ekf_results[last_ep]['X'].copy()
    Ps = ekf_results[last_ep]['P'].copy()
    
    rts_results[last_ep] = {
        'xyz': np.array([Xs[0], Xs[2], Xs[4]]),
        'lla': xyz2lla(np.array([Xs[0], Xs[2], Xs[4]])),
        'pr_residuals': ekf_results[last_ep].get('pr_residuals', np.array([])),
    }
    
    # Backward pass
    for k in range(len(valid_epochs) - 2, -1, -1):
        ep = valid_epochs[k]
        ep_next = valid_epochs[k + 1]
        
        dt = ep_next - ep
        if dt <= 0:
            dt = 1.0
        
        Xhat_k = ekf_results[ep]['X']
        Phat_k = ekf_results[ep]['P']
        
        # State transition matrix
        a = np.array([[1, dt], [0, 1]])
        a0 = np.zeros((2, 2))
        A = np.block([
            [a, a0, a0, a0],
            [a0, a, a0, a0],
            [a0, a0, a, a0],
            [a0, a0, a0, a]
        ])
        
        # Predicted state at k+1
        Xp_kp1 = A @ Xhat_k
        Pp_kp1 = A @ Phat_k @ A.T  # approximate (ignoring Q for simplicity)
        
        # Add small regularization if needed
        try:
            Pp_inv = inv(Pp_kp1 + np.eye(8) * 1e-6)
        except np.linalg.LinAlgError:
            Pp_inv = pinv(Pp_kp1)
        
        # Smoothing gain
        G = Phat_k @ A.T @ Pp_inv
        
        # Smoothed state
        Xs = Xhat_k + G @ (Xs - Xp_kp1)
        Ps = Phat_k + G @ (Ps - Pp_kp1) @ G.T
        
        xyz_s = np.array([Xs[0], Xs[2], Xs[4]])
        rts_results[ep] = {
            'xyz': xyz_s,
            'lla': xyz2lla(xyz_s),
            'pr_residuals': ekf_results[ep].get('pr_residuals', np.array([])),
        }
    
    return rts_results


# ============================================================
# MHE - Moving Horizon Estimator (from MHEstimator.m)
# ============================================================

def run_mhe(epoch_data_dict, sorted_epochs, window_size=8):
    """
    Moving Horizon Estimator.
    At each epoch, uses a window of past measurements to solve a batch WLS problem.
    Returns dict of epoch -> {'xyz': ..., 'lla': ...}
    """
    results = {}
    
    # Initial state
    xo = np.zeros(8)
    xo_initialized = False
    
    for idx, ep in enumerate(sorted_epochs):
        ed = epoch_data_dict[ep]
        num_sv = len(ed['prns'])
        
        if num_sv < 4:
            continue
        
        if not xo_initialized:
            xo[:3] = ed['wls_xyz']
            xo[3] = ed['wls_bc']
            xo_initialized = True
        
        # Determine window
        start_idx = max(0, idx - window_size)
        window_epochs = sorted_epochs[start_idx:idx + 1]
        
        # Collect all measurements in window
        all_sv_xyz = []
        all_pr = []
        all_pr_sigma = []
        
        for w_ep in window_epochs:
            w_ed = epoch_data_dict.get(w_ep)
            if w_ed is None or len(w_ed['prns']) < 4:
                continue
            all_sv_xyz.append(w_ed['sv_xyz'])
            all_pr.append(w_ed['raw_pr'])
            sigma = np.clip(w_ed['pr_error_unc'], 1.0, 100.0)
            all_pr_sigma.append(sigma)
        
        if len(all_sv_xyz) == 0:
            continue
        
        sv_xyz_all = np.vstack(all_sv_xyz)
        pr_all = np.concatenate(all_pr)
        sigma_all = np.concatenate(all_pr_sigma)
        
        total_sv = len(pr_all)
        if total_sv < 4:
            continue
        
        # Iterative WLS
        xyz_est = xo[:3].copy()
        bc_est = xo[3]
        
        for iteration in range(5):
            dxyz = xyz_est[np.newaxis, :] - sv_xyz_all
            ranges = np.sqrt(np.sum(dxyz ** 2, axis=1))
            
            # Unit vectors
            los = dxyz / ranges[:, np.newaxis]
            
            # Design matrix
            H = np.column_stack([los, np.ones(total_sv)])
            
            # Residuals
            pr_pred = ranges + bc_est
            dz = pr_all - pr_pred
            
            # Weighted least squares
            W = np.diag(1.0 / sigma_all)
            try:
                WH = W @ H
                dx = inv(WH.T @ WH) @ WH.T @ (W @ dz)
            except np.linalg.LinAlgError:
                dx = pinv(W @ H) @ (W @ dz)
            
            xyz_est += dx[:3]
            bc_est += dx[3]
            
            if norm(dx[:3]) < 0.01:
                break
        
        xo[:3] = xyz_est
        xo[3] = bc_est
        
        lla = xyz2lla(xyz_est)
        
        # Compute pseudorange residuals
        ranges_final = np.sqrt(np.sum((ed['sv_xyz'] - xyz_est[np.newaxis, :]) ** 2, axis=1))
        pr_res = ed['raw_pr'] - (ranges_final + bc_est)
        
        results[ep] = {
            'xyz': xyz_est.copy(),
            'lla': lla,
            'pr_residuals': pr_res,
        }
    
    return results


# ============================================================
# Apply PrNet corrections
# ============================================================

def apply_prnet_corrections(epoch_data_dict, prnet_bias, sorted_epochs):
    """
    Apply PrNet-predicted pseudorange corrections to raw pseudoranges.
    
    prnet_bias: Nx5 array from PrNet output [epoch, prn, predicted_bias, ...]
    Returns corrected epoch_data_dict.
    """
    corrected = {}
    
    # Build lookup: (epoch, prn) -> correction
    correction_map = {}
    if prnet_bias is not None and len(prnet_bias) > 0:
        for row in prnet_bias:
            ep = int(row[0])
            prn = int(row[1])
            corr = row[2]
            correction_map[(ep, prn)] = corr
    
    for ep in sorted_epochs:
        ed = epoch_data_dict[ep]
        corrected_ed = dict(ed)  # shallow copy
        
        # Apply corrections
        corrected_pr = ed['raw_pr'].copy()
        for i, prn in enumerate(ed['prns']):
            key = (ep, prn)
            if key in correction_map:
                corrected_pr[i] -= correction_map[key]
        
        corrected_ed['raw_pr'] = corrected_pr
        corrected[ep] = corrected_ed
    
    return corrected


# ============================================================
# Compute position errors
# ============================================================

def compute_position_errors(results, epoch_data_dict, sorted_epochs):
    """
    Compute horizontal position errors for each epoch.
    Returns arrays of (epoch, error_m).
    """
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
