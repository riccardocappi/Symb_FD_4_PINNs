from models.pinn import MLP, activations
from .experiment import Experiments
import torch
from utils.utils import get_pysr_model, symbolic_regression, make_callable, parse_pysr_equation
import sympy as sp
from models.pinn import LWR_NN
import logging
import os
import json
from utils.utils import poly_fit_greenshields


class ExperimentSymbFD(Experiments):
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
        load_data=False, 
        data_name="", 
        sample_ratio=0.75,
        vel_data_path = "./data/I80/NGSIM_US80_4pm_Velocity_Data.txt",
        dens_data_path = "./data/I80/NGSIM_US80_4pm_Density_Data.txt",
        poly_fit = False,
        ablation=False,
        pysr_ms = "loss"
    ):
        super().__init__(config, n_trials, model_selection_method, study_name, process_id, val_perc, seed, n_spacing, m_range, scale, verbose, load_data, data_name, 
                         sample_ratio, vel_data_path=vel_data_path, dens_data_path=dens_data_path)
        
        self.poly_fit = poly_fit
        self.ablation = ablation
        if not poly_fit:
            self.fd_func_loss, sympy_func_loss, self.params_loss = self._get_symb_fd(model_selection=pysr_ms)
            self.params_loss["function"] = str(sympy_func_loss)
            os.makedirs(f"{self.model_path}/symb_fds", exist_ok=True)
            with open(f"{self.model_path}/symb_fds/params_loss.json", "w") as f:
                json.dump(self.params_loss, f)
                
            self.rho_max = self._get_rho_max(sympy_func_loss)
        else:
            self.power_to_func = {}
        
        
    
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
        
        if not self.poly_fit:
            symb_fd = self.fd_func_loss
            rho_max = self.rho_max
        else:
            power = trial.suggest_int(
                "power",
                self.search_space["power"][0],
                self.search_space["power"][-1]
            )
            
            symb_fd, rho_max = self._get_poly_fit(power)
            
            
            
        model = LWR_NN(
            main_model=main_model,
            fd_func=symb_fd,
            loss_fn_deep=loss_fn_deep,
            loss_fn_phys=loss_fn_phy,
            fd = "symb",
            predict_velocity=True,
            mu_1=mu1,
            mu_2=mu2,
            rho_max_init=rho_max,
            v_max_init=None,
            split_out=False,
            ablation=self.ablation
        )
        
        model = model.to(torch.device(self.device))
        
        return model
        

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
        
        out_dim = 1
        
        hidden_layers = [n_hidden_neurons] * n_hidden_layers
        hidden_layers = [2] + hidden_layers + [out_dim]
         
        af_str = activation_function
        assert af_str in activations, f"Activation function '{af_str}' not recognized."
        af = activations[af_str]
        dropout_rate = 0.0
        save_black_box = False
        
        mlp = MLP(hidden_layers, af, dropout_rate, save_black_box)
        
        return mlp
    

    def _get_symb_fd(self, model_selection):
        
        x = self.dens_train[:, 0].detach().cpu().numpy()
        y = self.u_train[:, 0].detach().cpu().numpy()
        pysr = lambda: get_pysr_model(n_iterations=100, parallelism="serial", random_state = self.seed,  deterministic = True)
        
        top5_symb_fds = symbolic_regression(x.reshape(-1, 1), y.reshape(-1, 1), pysr, model_selection=model_selection)
        # dens_to_vel = sp.sympify(top5_symb_fds.iloc[0]["sympy_format"])
        dens_to_vel, params = parse_pysr_equation(top5_symb_fds.iloc[0]["equation"])
        logging.info(f"Best symbolic equation according to {model_selection}: {dens_to_vel}")
        print(f"Best symbolic equation according to {model_selection}: {dens_to_vel}")
        
        flux_fun = sp.symbols('x0') * dens_to_vel
        
        symb_torch = make_callable(flux_fun)
                
        return symb_torch, dens_to_vel, params
    
    def _get_rho_max(self, eq):
        def min_positive_solution(var):
            # Solve equation
            sols = sp.solve(eq, var)

            positive_real_sols = []

            for s in sols:
                # Try to evaluate numerically
                s_eval = sp.N(s)
                if s_eval.is_real and s_eval > 0:
                    positive_real_sols.append(s_eval)

            if not positive_real_sols:
                return None

            return min(positive_real_sols)
        
        x0 = sp.Symbol('x0')
        try:
            sol = min_positive_solution(x0)
        except Exception as e:
            default_bound = self.dens_train.max().item() + (self.dens_train.max().item() * 0.05)
            logging.warning(f"Could not solve eq. {str(eq)} for rho_max, returning default bound = {default_bound}")
            return default_bound 
        
        if sol is None:
            default_bound = self.dens_train.max().item() + (self.dens_train.max().item() * 0.05)
            logging.warning(f"No positive real solution found for eq. {str(eq)}, returning default bound = {default_bound}")
            return default_bound
        
        print(f"Positive real solution for eq. {str(eq)}: {sol}")
        logging.info(f"Positive real solution for eq. {str(eq)}: {sol}")
        
        return float(sol)
    

    def _get_poly_fit(self, power):
        if power not in self.power_to_func:
            _, dens_to_vel = poly_fit_greenshields(
                dens_train=self.dens_train[:, 0].detach().cpu().numpy(),
                u_train=self.u_train[:, 0].detach().cpu().numpy(),
                power = power
            )
            
            flux_fun = sp.symbols('x0') * dens_to_vel
            rho_max = self._get_rho_max(dens_to_vel)
            
            self.power_to_func[power] = (flux_fun, rho_max, dens_to_vel)
        
        
        flux_fun = self.power_to_func[power][0]
        symb_fd = make_callable(flux_fun)
        rho_max = self.power_to_func[power][1]
        dens_to_vel = self.power_to_func[power][2]
        logging.info(f"Best poly fitting with power={power}: {dens_to_vel}")
        print(f"Best poly fitting with power={power}: {dens_to_vel}")
        return symb_fd, rho_max
        
        
        
        
