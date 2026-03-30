from abc import ABC, abstractmethod
import torch
import os
import logging
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import yaml
from optuna.samplers import GridSampler
import json
from train_loop import seed_everything, train_loop
from utils.data_utils import build_real_dataset
from torch.utils.data import TensorDataset
from typing import Optional
from models.pinn import PINN
import matplotlib.pyplot as plt
import numpy as np
from tsl.data.preprocessing.scalers import MinMaxScaler


SCORES = {
    'MSE': torch.nn.MSELoss(),
    'MAE': torch.nn.L1Loss()
}

def mae_metric(pred: torch.Tensor, target: torch.Tensor):
    return torch.mean(torch.abs(pred - target))

def mse_metric(pred: torch.Tensor, target: torch.Tensor):
    return torch.mean((pred - target) ** 2)

def rmse_metric(pred: torch.Tensor, target: torch.Tensor):
    return torch.sqrt(torch.mean((pred - target) ** 2))

def relative_rmse_metric(pred: torch.Tensor, target: torch.Tensor):
    mse = torch.mean((pred - target) ** 2)
    rmse = torch.sqrt(mse)
    relative_rmse = rmse / torch.sqrt(torch.mean(target ** 2))
    return relative_rmse

def plot_heatmap(X, T, Grid, title='Heatmap', vmin = 0, vmax = 25, savefig=True, save_path="./heatmap.png"):

    plt.figure(figsize=(8, 6))

    plt.pcolormesh(
        X, T, Grid,
        shading='auto',
        cmap='jet',
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar(label='v(x,t)')
    plt.xlabel('Location x (m)')
    plt.ylabel('Time t (s)')
    plt.title(title)

    plt.tight_layout()
    if savefig:
        plt.savefig(save_path)
        
    
    
def delete_study(study_name, storage = "journal"):
    assert (storage == "journal") or (storage == "sqlite"), "Not supported storage backend!"
    storage_backend = "sqlite:///optuna_study.db" if storage == "sqlite" else JournalStorage(JournalFileBackend("optuna_journal_storage.log"))
    
    optuna.delete_study(
        study_name=study_name,
        storage=storage_backend
    )

def delete_all_studies(storage_type="journal"):
    storage = "sqlite:///optuna_study.db" if storage_type == "sqlite" else JournalStorage(JournalFileBackend("optuna_journal_storage.log"))
    study_summaries = optuna.get_all_study_summaries(storage=storage)

    # Delete each study
    for summary in study_summaries:
        print(summary.study_name)
        optuna.delete_study(study_name=summary.study_name, storage=storage)


class Experiments(ABC):
    def __init__(
        self, 
        config, 
        n_trials,
        model_selection_method='optuna',
        study_name='example',
        process_id=0,
        val_perc=0.2,
        seed=42,
        n_spacing=100,
        m_range=10,        
        scale=True,
        verbose = False,
        load_data = False,
        data_name="",
        sample_ratio = 0.75,
        vel_data_path = "./data/I80/NGSIM_US80_4pm_Velocity_Data.txt",
        dens_data_path = "./data/I80/NGSIM_US80_4pm_Density_Data.txt"
    ):
        assert model_selection_method != 'optuna' or n_trials is not None
        assert model_selection_method == 'optuna' or model_selection_method == 'grid_search', 'Optimization method not supported!'
        
        self.config = config
        self.n_trials = n_trials
        self.method = model_selection_method
        
        self.device = config.get("device", "cuda:0")
        if self.device == 'cuda':
            assert torch.cuda.is_available()      
        
        save_dir = f"./data/torch_data"
        os.makedirs(save_dir, exist_ok=True)
        
        if not load_data:
            X_u_train, u_train, dens_train, train_idx, X_f_train, X_all, u_all, X, T, Grid = self._generate_data(
                vel_data_path=vel_data_path,
                dens_data_path=dens_data_path,
                seed = seed,
                sample_ratio=sample_ratio,
                n_spacing=n_spacing,
                m_range=m_range
            )
            if data_name != "":
                data_all = {
                    "X_u_train": X_u_train,
                    "u_train": u_train,
                    "dens_train": dens_train,
                    "train_idx": train_idx,
                    "X_f_train": X_f_train,
                    "X_all": X_all,
                    "u_all": u_all,
                    "X": X,
                    "T": T,
                    "Grid": Grid
                }
                torch.save(data_all, f"{save_dir}/{data_name}")
        else:
            assert data_name != ""
            data_all = torch.load(f"{save_dir}/{data_name}", weights_only=False)
            X_u_train, u_train = data_all["X_u_train"], data_all["u_train"]
            dens_train = data_all["dens_train"]
            train_idx = data_all["train_idx"]
            X_f_train = data_all["X_f_train"]
            X_all, u_all = data_all["X_all"], data_all["u_all"]
            X, T = data_all["X"], data_all["T"]
            Grid = data_all["Grid"]
            
            print("Data Loaded!")
            
            
        self.dataset = TensorDataset(X_u_train, u_train)
        self.X_f_train = X_f_train
        self.dens_train = dens_train
        self.u_train = u_train
        
        X_all = X_all.to(self.device)
        u_all = u_all.to(self.device)
        
        self.test_set = (X_all, u_all)
        
        # To plot learned field
        self.Grid = Grid
        self.X = X
        self.T = T
        self.train_idx = train_idx.to(self.device)
        
        self.val_perc = val_perc
        
        self.verbose = verbose
        
        self.opt = config["opt"]
        self.log = config.get("log", 10)
        
        self.process_id = process_id
        self.model_path = f'./trained_models/{config["model_name"]}/{study_name}/{str(process_id)}'
        
        self.search_space = config["search_space"]
        self.seed = seed
        
        criterion_str = config.get("criterion", "MSE")
        assert criterion_str in SCORES, "Not supported criterion!"
        
        self.criterion = SCORES[config.get('criterion', 'MSE')]
        self.study_name = f'{self.config["model_name"]}-{study_name}'
        logs_folder = f'{self.model_path}/optuna_logs'
        
        os.makedirs(logs_folder, exist_ok=True)
        logs_file_path = f'{logs_folder}/optuna_logs.txt'
        logger = logging.getLogger()
        
        logger.setLevel(logging.INFO)  # Setup the root logger.
        self.optuna_handler = logging.FileHandler(logs_file_path, mode="a")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.optuna_handler.setFormatter(formatter)
        
        logger.addHandler(self.optuna_handler)
        optuna.logging.enable_propagation()

        self.current_ckpt = None
        self.current_model_arch = None # Store model architecture
        self.best_model: Optional[PINN] = None
        
        storage = self.config.get('storage', 'journal')
        assert (storage == "journal") or (storage == "sqlite"), "Not supported storage backend!"
        
        self.storage = "sqlite:///optuna_study.db" if storage == "sqlite" else JournalStorage(JournalFileBackend("optuna_journal_storage.log"))
        self.scale = scale
        
        self.scaler = None
        
        copy_config_path = f'{self.model_path}/config.yml'
        if not os.path.exists(copy_config_path):
            with open(copy_config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
                
    
    def run(self):
        
        seed_everything(self.seed)
        
        if self.scale:
            self.scaler = self.pre_processing()
            
        self.optimize() # Optuna study optimization 

        self.test_evaluation()
        
        logging.getLogger().removeHandler(self.optuna_handler)
        optuna.logging.disable_propagation()
        
    
    def optimize(self):
        if self.method == 'grid_search':
            sampler = GridSampler(self.search_space)
            n_trials = len(sampler._all_grids)
        else:
            sampler = optuna.samplers.TPESampler(seed=self.seed)
            n_trials = self.n_trials
            
        study = optuna.create_study(
            direction='minimize',
            study_name=self.study_name,
            sampler=sampler,
            storage=self.storage,
            load_if_exists=True
        )
        
        study.optimize(
            self.objective, 
            n_trials=n_trials, 
            callbacks=[self.callback]
        )
        
    
    def objective(self, trial):
        lr_space = self.search_space.get('lr', [0.001])
        lr = trial.suggest_float('lr', lr_space[0], lr_space[-1], log=True)
        
        batch_size_space = self.search_space.get('batch_size', [-1])
        batch_size = trial.suggest_categorical('batch_size', batch_size_space)
        
        epochs_search_space = self.search_space.get("epochs", [100])
        epochs = trial.suggest_int("epochs", epochs_search_space[0], epochs_search_space[-1])
        
        patience_search_space = self.search_space.get("patience", [20])
        patience = trial.suggest_int("patience", patience_search_space[0], patience_search_space[-1])
        
        model = self.get_model_opt(trial)
        
        best_ckpt = train_loop(
            model = model,
            dataset = self.dataset,
            X_f_train=self.X_f_train,
            lr=lr,
            n_epochs=epochs,
            batch_size=batch_size,
            device=self.device,
            metric = self.criterion,
            val_split=self.val_perc,
            patience=patience,
            scaler=self.scaler,
            verbose=self.verbose
        )
        
        results = best_ckpt["results"]
        trial.set_user_attr("process_id", self.process_id)
        best_val_loss = min(results['val_loss'])
        self.current_ckpt = best_ckpt
        self.current_model_arch = model
        
        return best_val_loss
    
    
    def callback(self, study, trial):

        best_params = study.best_params
        if study.best_trial == trial:
            self._save_ckpt(self.current_ckpt, best_params)
            self.best_model = self.current_model_arch
        
        self.current_ckpt = None
        self.current_model_arch = None
            
    
    @abstractmethod
    def get_model_opt(self, trial):
        raise NotImplementedError()
    
    
    def pre_processing(self):
        X_star = torch.from_numpy(np.hstack((self.X.flatten()[:,None], self.T.flatten()[:,None])))
        scaler = MinMaxScaler(axis = 0, out_range=(-1, 1))
        scaler.fit(X_star)
        scaler.scale = scaler.scale.to(self.device)
        scaler.bias = scaler.bias.to(self.device)
        
        return scaler
        
        

    def test_evaluation(self):
        if os.path.exists(f"{self.model_path}/ckpt.pth") and (self.best_model is not None):
            ckpt = torch.load(f"{self.model_path}/ckpt.pth", map_location=torch.device(self.device), weights_only=False)
            
            assert 'model_state_dict' in ckpt, "No model state dict found in checkpoint!"
            
            self.best_model.load_state_dict(ckpt['model_state_dict'])
            
            X_all, u_all = self.test_set 
            self.best_model = self.best_model.eval()
            
            all_idx = torch.arange(X_all.shape[0], device=self.device)
            test_idx = all_idx[~torch.isin(all_idx, self.train_idx)]
                        
            if self.scaler is not None:
                X_in = self.scaler.transform(X_all)
            else:
                X_in = X_all
            
            with torch.no_grad():
                u_pred_all = self.best_model.get_preds(X_in)
            
            u_test = u_all[test_idx]
            u_pred_test = u_pred_all[test_idx]
            
            # err = self.criterion(u_pred_test, u_test).item()
            mse_err = mse_metric(u_pred_test, u_test).item()
            mae_err = mae_metric(u_pred_test, u_test).item()
            rmse_err = rmse_metric(u_pred_test, u_test).item()
            relative_rmse_metric_err = relative_rmse_metric(u_pred_test, u_test).item()
            
            ckpt["test_mse"] = mse_err
            ckpt["test_mae"] = mae_err  
            ckpt["test_rmse"] = rmse_err
            ckpt["test_relative_rmse"] = relative_rmse_metric_err
            
            print(f"Test MSE: {mse_err}")
            logging.info(f"Test MSE: {mse_err}")
            print(f"Test MAE: {mae_err}")
            logging.info(f"Test MAE: {mae_err}")
            print(f"Test RMSE: {rmse_err}")
            logging.info(f"Test RMSE: {rmse_err}")
            print(f"Test Relative RMSE: {relative_rmse_metric_err}")
            logging.info(f"Test Relative RMSE: {relative_rmse_metric_err}")
            
            ckpt["test_pred"] = u_pred_test.detach().cpu()
            ckpt["test_true"] = u_test.detach().cpu()
            
            self._save_ckpt(ckpt, None)
            
            os.makedirs(f"{self.model_path}/figs", exist_ok=True)
            
            u_pred_all = u_pred_all.detach().cpu().numpy().reshape(self.Grid.shape)
            vmin, vmax = np.min(self.Grid), np.max(self.Grid)
            
            plot_heatmap(
                X = self.X,
                T = self.T,
                Grid = u_pred_all,
                title = "Estimated field",
                vmin = vmin,
                vmax = vmax,
                savefig=True,
                save_path=f"{self.model_path}/figs/heatmap.png"
            )
            

    def _save_ckpt(self, ckpt, best_params=None):
        torch.save(ckpt, f"{self.model_path}/ckpt.pth")
        
        if best_params is not None:
            with open(f"{self.model_path}/best_params.json", "w") as f:
                json.dump(best_params, f)
                
    
    def _generate_data(self, vel_data_path, dens_data_path, seed, sample_ratio, n_spacing=100, m_range=10):
        X_u_train, u_train, other_train, train_idx ,X_f_train, X_all, u_all, X, T, Grid, _ = build_real_dataset(
            data_path = vel_data_path,
            other_data_path= dens_data_path,
            seed = seed,
            n_spacing=n_spacing,
            m_range=m_range,
            sample_ratio=sample_ratio
        )
        
        return X_u_train, u_train, other_train, train_idx ,X_f_train, X_all, u_all, X, T, Grid
        