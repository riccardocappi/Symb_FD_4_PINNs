from .experiment import Experiments
import torch
from models.pinn import LWR_NN
from models.pinn import activations, MLP
from utils.utils import estimate_greenshields_params


class ExperimentPINN(Experiments):
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
        learn_fd = False,
        scale=True,
        split_out = False,
        verbose = False,
        load_data = False,
        data_name="",
        sample_ratio = 0.75,
        vel_data_path = "./data/I80/NGSIM_US80_4pm_Velocity_Data.txt",
        dens_data_path = "./data/I80/NGSIM_US80_4pm_Density_Data.txt",
        end_to_end = False,
        ablation = False
    ):
        self.learn_fd = learn_fd
        self.split_out = split_out
        super().__init__(config, n_trials, model_selection_method, study_name, process_id, val_perc, seed, n_spacing, m_range, scale=scale, verbose=verbose,
                         load_data=load_data, data_name=data_name, sample_ratio=sample_ratio, vel_data_path=vel_data_path, dens_data_path=dens_data_path)
        
        
        params = estimate_greenshields_params(
            self.u_train[:, 0].detach().cpu().numpy(),
            self.dens_train[:, 0].detach().cpu().numpy()
        )
        
        self.v_max = params[0]
        self.rho_max = params[1]
        self.end_to_end = end_to_end
        self.ablation = ablation
        
    
    def get_model_opt(self, trial):
        main_model = self._get_main_model(trial)
        
        mu1 = trial.suggest_float(
            "mu1",
            self.search_space["mu1"][0],
            self.search_space["mu1"][-1]
        )
        
        mu2 = trial.suggest_float(
            "mu2",
            self.search_space["mu2"][0],
            self.search_space["mu2"][-1]
        )
        
        loss_fn_deep = torch.nn.MSELoss()
        loss_fn_phy = torch.nn.MSELoss()
        
        fd_func = None
        xi = 0.0
        r_a = 0.0
        r_b = 0.0
        if self.learn_fd:
            fd_func = self._get_fd_learner(trial)
            xi = trial.suggest_float(
                "xi",
                self.search_space["xi"][0],
                self.search_space["xi"][-1]
            )
            a_b_bound = trial.suggest_categorical("a_b_bound", [0.05, 0.10, 0.15])

            dens_max = self.dens_train.max().item()

            r_a = dens_max * (1 - a_b_bound)
            r_b = dens_max * (1 + a_b_bound)

            print(f"a_b_bound: {a_b_bound*100:.0f}% | r_a: {r_a}, r_b: {r_b}")
            
        
        model = LWR_NN(
            main_model=main_model,
            fd_func=fd_func,
            loss_fn_deep=loss_fn_deep,
            loss_fn_phys=loss_fn_phy,
            fd = "greenshields" if not self.learn_fd else "learned",
            predict_velocity=True,
            mu_1=mu1,
            mu_2=mu2,
            rho_max_init=self.rho_max,
            v_max_init=self.v_max,
            split_out=self.split_out,
            xi = xi,
            end_to_end=self.end_to_end,
            r_a = r_a,
            r_b = r_b,
            ablation=self.ablation
        )
        
        model = model.to(torch.device(self.device))
        
        return model
        
    
    def _get_fd_learner(self, trial):
        n_hidden_neurons_fd = trial.suggest_int(
            "n_hidden_neurons_fd",
            self.search_space["n_hidden_neurons_fd"][0],
            self.search_space["n_hidden_neurons_fd"][-1]
        )
        
        n_hidden_layers_fd = trial.suggest_int(
            "n_hidden_layers_fd",
            self.search_space["n_hidden_layers_fd"][0],
            self.search_space["n_hidden_layers_fd"][-1]
        ) 
        
        activation_function_fd = trial.suggest_categorical("activation_function_fd", self.search_space["activation_function_fd"])
        
        hidden_layers = [n_hidden_neurons_fd] * n_hidden_layers_fd
        hidden_layers = [1] + hidden_layers + [1]
        
        af_str = activation_function_fd
        assert af_str in activations, f"Activation function '{af_str}' not recognized."
        af = activations[af_str]
        dropout_rate = 0.0
        save_black_box = False
        
        mlp = MLP(hidden_layers, af, dropout_rate, save_black_box)
        
        return mlp
    
    
    def _get_main_model(self, trial):
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
        
        out_dim = 1 if not self.split_out else 2
        
        hidden_layers = [n_hidden_neurons] * n_hidden_layers
        hidden_layers = [2] + hidden_layers + [out_dim]
        
        af_str = activation_function
        assert af_str in activations, f"Activation function '{af_str}' not recognized."
        af = activations[af_str]
        dropout_rate = 0.0
        save_black_box = False
        
        mlp = MLP(hidden_layers, af, dropout_rate, save_black_box)
        
        return mlp
        
        
        
        
        
        
        