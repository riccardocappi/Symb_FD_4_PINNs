from scipy.optimize import curve_fit
from pysr import PySRRegressor, TemplateExpressionSpec
import numpy as np
import torch
import sympytorch
import sympy as sp
import re


def estimate_greenshields_params(u_train, dens_train, init_params = [20.0, 1.0]):
    func_optim = lambda x, vmax, rhomax:  vmax - vmax * x / rhomax
    params, _ = curve_fit(func_optim, dens_train, u_train, p0=init_params, nan_policy='omit')
    
    return params


def poly_fit_greenshields(u_train, dens_train, power = 2, init_params = [20.0, 1.0]):
    assert power in [2, 3], f"Power {power} not supported!"
    x0 = sp.Symbol('x0')
    v_f_sp, p_j_sp = sp.symbols('v_f p_j')
    if power == 2:
        func_optim = lambda x, v_f, p_j: v_f * ( (1 - 2*x/p_j) + (x/p_j)**2)
        sym_func = v_f_sp * ((1 - 2*x0/p_j_sp) + (x0/p_j_sp)**2)
    else:
        func_optim = lambda x, v_f, p_j: v_f * ((1 - 3*x/p_j) + 3*(x/p_j)**2 - (x/p_j)**3)
        sym_func = v_f_sp * ((1 - 3*x0/p_j_sp) + 3*(x0/p_j_sp)**2 - (x0/p_j_sp)**3)
        
    params, _ = curve_fit(func_optim, dens_train, u_train, p0=init_params, nan_policy='omit')
    sym_fitted = sym_func.subs({v_f_sp: params[0], p_j_sp: params[1]})
    
    return params, sp.simplify(sym_fitted)


def get_pysr_model(
    n_iterations=100,
    binary_operators = ['+', '-', '*', '/'], 
    extra_sympy_mappings = {},
    unary_operators = None,
    **kwargs):
    
    extra_mapping = {"zero": lambda x: x*0}
    extra_mapping.update(extra_sympy_mappings)
    
    if unary_operators is None:
        unary_operators = [
            "neg",
            "square",
            "cube",
            "zero(x) = 0*x"
        ]
    
    expression_spec = TemplateExpressionSpec(
        expressions=["g"],
        parameters={"vmax": 1, "rhomax": 1},
        variable_names=["x"],
        combine="""
            y1 = vmax[1] - vmax[1] * x / rhomax[1]
            g(x, y1)
        """
    )
    
    model = PySRRegressor(
        niterations=n_iterations,  # Number of iterations
        unary_operators=unary_operators,
        binary_operators=binary_operators,
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        maxsize=7,
        maxdepth=5,
        verbosity=0,
        extra_sympy_mappings=extra_mapping,
        delete_tempfiles=True,
        temp_equation_file=True,
        tempdir='./pysr',
        progress=False,
        expression_spec=expression_spec,
        **kwargs
    )
    
    return model



def symbolic_regression(x, y, pysr_model = None, sample_size = -1, seed=42, model_selection="loss"):
    rng = np.random.default_rng(seed)
    if pysr_model is None:
        model = get_pysr_model()
    else:
        model = pysr_model()
    
    if sample_size > 0 and sample_size < len(x):
        indices = rng.choice(len(x), sample_size, replace=False)
        x_sampled = x[indices]
        y_sampled = y[indices]
    else:
        x_sampled = x
        y_sampled = y
    
    model.fit(x_sampled, y_sampled)
    
    # return model.sympy()
    if model_selection == "score":
        top_5_eq = model.equations_.nlargest(5, model_selection)
    elif model_selection == "loss":
        top_5_eq = model.equations_.nsmallest(5, model_selection)
    else:
        raise ValueError("Model selection not recognized")
    return top_5_eq


class symb_wrapper(torch.nn.Module):
    def __init__(self, symb):
        super().__init__()
        self.symb = symb
    
    def forward(self, x):
        return self.symb(x0=x[:, 0])


def make_callable(expr):
    free_syms = expr.free_symbols
    if not free_syms:
        # Expression is constant
        const_value = float(expr)
        return lambda x: torch.full((x.shape[0], 1), const_value, dtype=x.dtype, device=x.device)

    expr = sympytorch.hide_floats(expr) # Doesn't train expr parameters
    sym_module = sympytorch.SymPyModule(expressions=[expr])
    
    syms = {str(s) for s in free_syms}
    if 'x0' in syms:
        return symb_wrapper(sym_module)
    else:
        raise ValueError(f"Unexpected symbols in expression: {free_syms}")
    

def extract_parameters(eq_str):
    pattern = r"(\w+)\s*=\s*\[([^\]]+)\]"
    matches = re.findall(pattern, eq_str)

    params = {}
    for name, value in matches:
        params[name] = float(value)

    return params

def extract_definitions(eq_str):
    parts = eq_str.split(";")
    defs = {}

    for part in parts:
        part = part.strip()
        if "=" not in part:
            continue
        name, rhs = part.split("=", 1)
        name = name.strip()
        rhs = rhs.strip()

        # Skip parameter blocks
        if rhs.startswith("["):
            continue

        defs[name] = rhs

    return defs

def pysr_rhs_to_sympy(rhs, arg_symbols):
    
    PYSR_TO_SYMPY = {
        "square": lambda a: a**2,
        "cube": lambda a: a**3,
        "neg": lambda a: -a,
        "abs": sp.Abs,
        "exp": sp.exp,
        "sin": sp.sin,
        "tan": sp.tan,
        "tanh": sp.tanh,
        "log": sp.log,
        "log1p": lambda a: sp.log(1 + a),
    }
    
    expr = rhs
    for i, sym in enumerate(arg_symbols, start=1):
        expr = expr.replace(f"#{i}", str(sym))

    return sp.sympify(expr, locals={**PYSR_TO_SYMPY})


def build_symbolic_functions(defs):

    x, y1 = sp.symbols("x0 y1")

    g_expr = pysr_rhs_to_sympy(defs["g"], [x, y1])

    return g_expr

def apply_combine(g_expr, vmax, rhomax):
    x, y1 = sp.symbols("x0 y1")
    y1_expr = vmax - vmax * x / rhomax
    
    final_expr = g_expr.subs({x: x, y1: y1_expr})

    return final_expr


def parse_pysr_equation(eq_str):
    params = extract_parameters(eq_str)
    defs = extract_definitions(eq_str)

    g_expr = build_symbolic_functions(defs)

    full_expr = apply_combine(
        g_expr,
        params["vmax"],
        params["rhomax"],
        # alpha=params["alpha"]
    )
    
    return full_expr, params


