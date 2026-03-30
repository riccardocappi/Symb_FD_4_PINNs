import scipy
import numpy as np
import matplotlib.pyplot as plt
import random
from pyDOE import lhs
import torch
import pandas as pd


def build_real_dataset(data_path, other_data_path, seed=1, n_spacing=500, m_range=10, sample_ratio=0.75):

    np.random.seed(seed)
    
    var = pd.read_table(data_path, delim_whitespace=True)
    other = pd.read_table(other_data_path, delim_whitespace=True)
    
    x_coords = np.arange(0, var.shape[0] * 20, 20)
    t_coords = np.arange(0, var.shape[1] * 5, 5)
    
    Exact = np.real(var.T)
    Exact_Other = np.real(other.T)
    
    X, T = np.meshgrid(x_coords, t_coords)
    
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]
    other_star = Exact_Other.flatten()[:, None]
    
    lb = X_star.min(0)
    ub = X_star.max(0)
    
    sensor_positions = np.arange(x_coords.min(), x_coords.max(), n_spacing)
    
    mask = np.zeros(X_star.shape[0], dtype=bool)
    
    for pos in sensor_positions:
        in_sensor_zone = np.abs(X_star[:, 0] - pos) <= m_range
        mask = mask | in_sensor_zone
    
    captured_indices = np.where(mask)[0]
    
    sample_size = int(len(captured_indices) * sample_ratio) 
    idx = np.random.choice(captured_indices, sample_size, replace=False)

    X_u_train = X_star[idx, :]
    u_train = u_star[idx, :]
    other_train = other_star[idx, :]
    
    all_points = Exact.shape[0] * Exact.shape[1] 
    N_f = int(0.8 * all_points)
    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    # X_f_train = np.vstack((X_f_train, X_u_train))
    # retain a maximum of 12000 points for X_f_train to manage computational load
    if X_f_train.shape[0] > 12000:
        idx_f = np.random.choice(X_f_train.shape[0], 12000, replace=False)
        X_f_train = X_f_train[idx_f, :]
    
    X_u_train_t = torch.tensor(X_u_train, dtype=torch.float32)
    u_train_t = torch.tensor(u_train, dtype=torch.float32)
    other_train_t = torch.tensor(other_train, dtype=torch.float32)
    X_f_train_t = torch.tensor(X_f_train, dtype=torch.float32)
    X_star_t = torch.tensor(X_star, dtype=torch.float32)
    u_star_t = torch.tensor(u_star, dtype=torch.float32)
    
    return X_u_train_t, u_train_t, other_train_t, torch.from_numpy(idx), X_f_train_t, X_star_t, u_star_t, X, T, Exact, other_star
    
    
        
    
