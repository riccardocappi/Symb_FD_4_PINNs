from .experiment import Experiments
from models.punn import PUNN
import torch


class ExperimentPUNN(Experiments):
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
        verbose=False,
        load_data = False,
        data_name="",
        sample_ratio = 0.75,
        vel_data_path = "./data/I80/NGSIM_US80_4pm_Velocity_Data.txt",
        dens_data_path = "./data/I80/NGSIM_US80_4pm_Density_Data.txt"
    ):
        super().__init__(config, n_trials, model_selection_method, study_name, process_id, val_perc, seed, n_spacing, m_range, scale, verbose=verbose,
                         load_data=load_data, data_name=data_name, sample_ratio=sample_ratio, vel_data_path=vel_data_path, dens_data_path=dens_data_path)
    
    
    def get_model_opt(self, trial):
        n_hidden_neurons = trial.suggest_int(
            "n_hidden_neurons",
            self.search_space["n_hidden_neurons"][0],
            self.search_space["n_hidden_neurons"][-1]
        )
        
        n_hidden_layers = trial.suggest_int(
            "n_hidden_layers",
            self.search_space["n_hidden_layers"][0],
            self.search_space["n_hidden_layers"][-1]
        ) 
        
        activation_function = trial.suggest_categorical("activation_function", self.search_space["activation_function"])
        
        model_conf = {
            "n_hidden_neurons": n_hidden_neurons,
            "n_hidden_layers": n_hidden_layers,
            "activation_function": activation_function,
        }
        
        loss_fn_deep = torch.nn.MSELoss()
        
        model = PUNN(
            model_conf=model_conf,
            loss_fn_deep=loss_fn_deep,
            input_dim=2,
            output_dim=1
        )
        
        model = model.to(torch.device(self.device))
        
        return model