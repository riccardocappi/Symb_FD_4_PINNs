import os
import argparse
import yaml
from experiments.experiment_pinn import ExperimentPINN
from experiments.experiment_punn import ExperimentPUNN
from experiments.experiment_symb_fd import ExperimentSymbFD

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def load_config(config_path='./conf/edenn.yml'):
    """
    Returns a dictionary of the specified config file
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    """
    Parse command line arguments for experiment configuration.
    """
    parser = argparse.ArgumentParser(description="Run STGNN Experiment with configurable parameters.")

    parser.add_argument(
        "--config_path",
        type=str,
        default="./conf/pinn.yml",
        help="Path to the configuration YAML file."
    )
    
    parser.add_argument(
        "--vel_path",
        type=str,
        default="./data/I80/NGSIM_US80_4pm_Velocity_Data.txt",
        help="Path to the velocity data."
    )
    
    parser.add_argument(
        "--dens_path",
        type=str,
        default="./data/I80/NGSIM_US80_4pm_Density_Data.txt",
        help="Path to the density data."
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of trials for model selection (e.g., Optuna)."
    )

    parser.add_argument(
        "--method",
        type=str,
        default="optuna",
        choices=["optuna", "grid_search"],
        help="Method used for model selection."
    )

    parser.add_argument(
        "--study_name",
        type=str,
        default="test",
        help="Name of the Optuna study or experiment run."
    )
    
    
    parser.add_argument(
        "--process_id",
        type=int,
        default=0,
        help="Process ID (useful for parallel runs)."
    )

    parser.add_argument(
        "--val_perc",
        type=float,
        default=0.2,
        help="Validation data percentage."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    
    parser.add_argument(
        "--learn_fd",
        action="store_true",
        help="Learn Fundamental Diagram"
    )
    
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Scale input data"
    )
    
    parser.add_argument(
        "--split_out",
        action="store_true",
        help=""
    )
    
    
    parser.add_argument(
        "--load_data",
        action="store_true",
        help="Load data"
    )
    
    parser.add_argument(
        "--data_name",
        type=str,
        default="test",
        help="Data name."
    )
    
    parser.add_argument(
        "--pysr_ms",
        type=str,
        default="loss",
        help="PySR model selction metric"
    )
    
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.75,
        help="Sampling ratio"
    )
    
    parser.add_argument(
        "--end_to_end",
        action="store_true",
        help="Train greenshields params end-to-end"
    )
    
    parser.add_argument(
        "--poly_fit",
        action="store_true",
        help="Fit a polynomial for symbolic FD"
    )
    
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Perform ablation without physics prior on rho_max"
    )
    
    
    
    return parser.parse_args()

def main():
    args = parse_args()
    conf = load_config(args.config_path)
    model_type = conf["model_name"]
    
    print(args)
    
    if model_type == "PINN":
        exp = ExperimentPINN(
            config=conf,
            n_trials=args.n_trials,
            model_selection_method=args.method,
            study_name=args.study_name,
            process_id=args.process_id,
            val_perc=args.val_perc,
            seed=args.seed,
            learn_fd=args.learn_fd,
            scale=args.scale,
            split_out=args.split_out,
            load_data=args.load_data,
            data_name=args.data_name,
            sample_ratio=args.sample_ratio,
            vel_data_path=args.vel_path,
            dens_data_path=args.dens_path,
            end_to_end=args.end_to_end,
            ablation=args.ablation
        )
    elif model_type == "PUNN":
        exp = ExperimentPUNN(
            config=conf,
            n_trials=args.n_trials,
            model_selection_method=args.method,
            study_name=args.study_name,
            process_id=args.process_id,
            val_perc=args.val_perc,
            seed=args.seed,
            scale=args.scale,
            load_data=args.load_data,
            data_name=args.data_name,
            sample_ratio=args.sample_ratio,
            vel_data_path=args.vel_path,
            dens_data_path=args.dens_path
        )
    elif model_type == "SYMB-FD":
        exp = ExperimentSymbFD(
            config=conf,
            n_trials=args.n_trials,
            model_selection_method=args.method,
            study_name=args.study_name,
            process_id=args.process_id,
            val_perc=args.val_perc,
            seed=args.seed,
            scale=args.scale,
            load_data=args.load_data,
            data_name=args.data_name,
            sample_ratio=args.sample_ratio,
            vel_data_path=args.vel_path,
            dens_data_path=args.dens_path,
            poly_fit=args.poly_fit,
            ablation=args.ablation,
            pysr_ms=args.pysr_ms
        )
    else:
        raise ValueError("Not implemented yet!")
    
    exp.run()
    

if __name__ == "__main__":
    main()