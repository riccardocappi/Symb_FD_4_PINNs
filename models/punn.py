
from models.pinn import PINN, MLP, activations
import torch
from typing import Callable

class PUNN(PINN):
    
    def __init__(
        self,
        model_conf:dict,
        loss_fn_deep:Callable = torch.nn.MSELoss(),
        input_dim:int = 2,
        output_dim:int = 1,
    ):
        super().__init__(
            loss_fn_deep=loss_fn_deep,
            loss_fn_phys=None
        )
        hidden_layers = [model_conf.get('n_hidden_neurons', 64)] * model_conf.get('n_hidden_layers', 1)
        hidden_layers = [input_dim] + hidden_layers + [output_dim]
        af_str = model_conf.get('activation_function', "relu")
        assert af_str in activations, f"Activation function '{af_str}' not recognized."
        af = activations[af_str]
        dropout_rate = model_conf.get('dropout_rate', 0.0)
        save_black_box = False
                
        self.mlp = MLP(hidden_layers, af, dropout_rate, save_black_box)
        
            
    
    def forward(self, x: torch.Tensor):
        raw_out = self.mlp(x)
        return raw_out
    
    
    def get_preds(self, x_train):
        return self(x_train)
    
    
    def fd(self, density: torch.Tensor):
        return 0
    
    
    def loss(self, x_train, u_train, X_f_train):
        y_hat = self.get_preds(x_train)
        loss = self.loss_fn_deep(y_hat, u_train)
        return loss