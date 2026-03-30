# from pysr import PySRRegressor
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from models.pinn import PINN
import random
from torch.utils.data import Subset, TensorDataset


def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)



def train_epoch(model:PINN, loader, X_f_train, optimizer, device, scaler = None):

    model = model.train()
    total_loss = 0.0
    total_samples = 0
            
    for x_train, u_train in loader:
        x_train = x_train.to(device)
        u_train = u_train.to(device)

        if scaler is not None:
            x_in = scaler.transform(x_train)
            X_f_in = scaler.transform(X_f_train)
        else:
            x_in = x_train
            X_f_in = X_f_train
        
        optimizer.zero_grad()

        loss = model.loss(x_in, u_train, X_f_in)

        # Backprop
        loss.backward()
        optimizer.step()

        batch_size = x_train.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def eval_model(model:PINN, loader, metric, device, scaler = None):

    model = model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_val, u_val in loader:
            x_val = x_val.to(device)
            u_val = u_val.to(device)
            
            if scaler is not None:
                x_in = scaler.transform(x_val)
            else:
                x_in = x_val
            
            y_pred = model.get_preds(x_in)
            
            loss = metric(y_pred, u_val)

            batch_size = x_val.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def train_loop(model:PINN, dataset, X_f_train, lr=1e-3, n_epochs=200, batch_size=32, device="cpu", metric = nn.MSELoss(), val_split=0.2,patience=20, scaler=None, verbose=False):
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, len(dataset)))
    
    batch_size_tr = batch_size if batch_size > 0 else len(train_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size_tr, shuffle=False) # Already shuffled
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    
    X_f_train = X_f_train.to(device)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_ckpt = {}
    
    pbar = tqdm(range(n_epochs), desc="Training Epochs", disable = (not verbose))
    
    results = defaultdict(list)
    
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, X_f_train, optimizer, device, scaler=scaler)
        val_loss = eval_model(model, val_loader, metric, device, scaler=scaler)
        
        if verbose:
            pbar.set_postfix(loss=train_loss, val_loss=val_loss)
        
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            ckpt = {
                'epoch': epoch,
                'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'val_loss': val_loss
            }
            best_model_ckpt = ckpt
        elif epoch - best_epoch >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch} with val loss {best_val_loss:.5e})")
            break
        
    if "model_state_dict" in best_model_ckpt:
        model.load_state_dict(best_model_ckpt['model_state_dict'])
    
    best_model_ckpt["results"] = results    
    return best_model_ckpt

    