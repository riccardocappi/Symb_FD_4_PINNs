import torch
from abc import ABC, abstractmethod
import torch.nn.functional as F
from typing import Callable
from typing import Union


activations = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "leaky_relu": F.leaky_relu,
    "softplus": F.softplus
}

class MLP(torch.nn.Module):
    """
    MLP Implementation
    """
    def __init__(self, hidden_layers, af, dropout_rate=0.0, save_black_box=False):
        super(MLP, self).__init__()
        self.af = af    # Activation function
        self.layers = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        
        for in_dim, out_dim in zip(hidden_layers, hidden_layers[1:]):
            self.layers.append(torch.nn.Linear(in_dim, out_dim))
            self.dropouts.append(torch.nn.Dropout(p=dropout_rate))
        
        self.cache_input = None
        self.cache_output = None
        self.save_black_box = save_black_box
    
    
    def forward(self, x: torch.Tensor):
        """
        MLP forward pass
        """
        if self.save_black_box:
            self.cache_input = x.detach()
        
        for i, (layer, dropout) in enumerate(zip(self.layers, self.dropouts)):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation and dropout except on the last layer
                x = self.af(x)
                x = dropout(x)
        
        if self.save_black_box:
            self.cache_output = x.detach()
        
        return x
    
    
class PINN(torch.nn.Module, ABC):
    """
    PINN model interface
    """
    def __init__(self, loss_fn_deep:Callable, loss_fn_phys:Callable):
        super(PINN, self).__init__()
        self.loss_fn_deep = loss_fn_deep
        self.loss_fn_phys = loss_fn_phys
    
    
    @abstractmethod
    def fd(self, density):
        pass
    
    @abstractmethod
    def loss(self, x_train, u_train, X_f_train):
        pass
    
    @abstractmethod
    def get_preds(self, x):
        pass
    
    
class LWR_NN(PINN):
    
    def __init__(
        self,
        main_model: MLP,
        fd_func: Union[MLP, Callable] = None,
        loss_fn_deep:Callable = torch.nn.MSELoss(),
        loss_fn_phys:Callable = torch.nn.MSELoss(),
        fd = "greenshields",
        predict_velocity:bool = True,
        mu_1:float = 0.5,
        mu_2:float = 0.5,
        rho_max_init = 0.15,
        v_max_init = 25.0,
        split_out = False,
        xi = 0.5,
        r_a = 0.0,
        r_b = 0.3,
        end_to_end=False,
        ablation = False
    ):
        super().__init__(
            loss_fn_deep=loss_fn_deep,
            loss_fn_phys=loss_fn_phys
        )
        
        assert fd in ["greenshields", "learned", "symb"], f"Fundamental diagram '{fd}' not recognized."
        
        if fd_func is not None:
            assert (fd == "learned") or (fd == "symb")
        
        self.fd_type = fd
        self.main_model = main_model
        self.ablation = ablation
        if self.ablation:
            print("Running ablation without physics prior on rho_max")
        
        self.predict_velocity = predict_velocity
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.xi = xi
        self.r_a = r_a
        self.r_b = r_b
        self.split_out = split_out
        
        if split_out:
            assert self.predict_velocity
        
        if fd == "greenshields":
            if not end_to_end:
                self.v_max = v_max_init
                self.rho_max = rho_max_init
            else:
                self.v_max = torch.nn.Parameter(torch.tensor(v_max_init, dtype=torch.float32))
                self.rho_max = torch.nn.Parameter(torch.tensor(rho_max_init, dtype=torch.float32))
        else:
            self.rho_max = rho_max_init
            self.fd_func = fd_func

    def forward(self, x: torch.Tensor):
        return self.main_model(x)
        
        
    def _get_dens(self, out):
        if (self.fd_type == "learned") or self.ablation:
            return out # To replicate results of PIDL-FD Paper
        else:
            out = self.rho_max * torch.sigmoid(out) # In the case of symbolic FD model, we want to ensure the density is between 0 and rho_max
        return out
        
    def get_preds(self, x):
        x_out = self(x)
        if not self.split_out: 
            x_dens = self._get_dens(x_out)
        else:
            x_vel = x_out[:, 0:1]
            return x_vel
        
        if self.predict_velocity:
            x_flux = self.fd(x_dens)
            y_hat = x_flux / (x_dens + 1e-6)
        else:
            y_hat = x_dens
        
        return y_hat
    
    
    def fd(self, density: torch.Tensor):
        if self.fd_type == "greenshields":
            v_max = self.v_max
            rho_max = self.rho_max   
            flux = v_max * density * (1.0 - density / rho_max)
        elif (self.fd_type == "symb") or (self.fd_type == "learned"):
            flux = self.fd_func(density)
        else:
            raise ValueError("Unknown fundamental diagram")

        return flux
    
    
    def loss(self, x_train, u_train, X_f_train):
        
        y_hat = self.get_preds(x_train)
        
        X_f_train.requires_grad_(True)
        
        f_out = self(X_f_train)
        
        if not self.split_out:
            f_dens = f_out
        else:
            f_dens = f_out[:, 1:2]
        
        f_dens = self._get_dens(f_dens)
        
        f_flux = self.fd(f_dens)
        
        f_dens_t = torch.autograd.grad(
            f_dens.sum(),
            X_f_train,
            create_graph=True
        )[0][:, 1:2]

        f_flux_x = torch.autograd.grad(
            f_flux.sum(),
            X_f_train,
            create_graph=True
        )[0][:, 0:1]
        
        f_pred = f_dens_t + f_flux_x
        
        base_loss = self.mu_1 * self.loss_fn_deep(y_hat, u_train) +  (self.mu_2 * self.loss_fn_phys(f_pred, torch.zeros_like(f_pred)))
        
        if (self.fd_type == "learned"):
            reg_loss = self.xi * self.compute_fd_concavity_reg(rho_min=self.r_a, rho_max=self.r_b, num_samples=300, device=f_out.device)
            loss = base_loss + reg_loss
        else:
            loss = base_loss
        
        return loss
    
    
    
    def compute_fd_concavity_reg(self, rho_min=0.0, rho_max=0.3, num_samples=100, device="cuda"):
        rho_samples = torch.linspace(rho_min, rho_max, num_samples, 
                                     device=device,
                                     requires_grad=True).unsqueeze(1)
        
        Q_samples = self.fd(rho_samples)
        
        dQ_drho = torch.autograd.grad(
            Q_samples.sum(),
            rho_samples,
            create_graph=True
        )[0]
        
        d2Q_drho2 = torch.autograd.grad(
            dQ_drho.sum(),
            rho_samples,
            create_graph=True
        )[0]

        penalty = torch.nn.functional.relu(d2Q_drho2)

        reg_loss = penalty.mean() * (rho_max - rho_min) # Rieman sum approx.
        
        return reg_loss