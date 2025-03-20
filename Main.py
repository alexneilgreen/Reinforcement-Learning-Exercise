import subprocess
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

def run_experiment(args_dict):
    """
    Run a single experiment with the specified parameters by calling RL_assignment.py
    
    Args:
        args_dict (dict): Dictionary containing the parameters for the experiment
    
    Returns:
        dict: The parameters used and the directory where results are stored
    """
    # Check if output directory already exists
    output_dir = args_dict["output_dir"]
    if os.path.exists(output_dir):
        print(f"Experiment results already exist in {output_dir}, skipping...")
        return args_dict
        
    # Convert dictionary to command-line arguments
    command = ["python", "RL_assignment.py"]
    
    for key, value in args_dict.items():
        command.append(f"--{key}")
        command.append(str(value))
    
    # Run the experiment
    print(f"Running experiment with: {' '.join(command)}")
    subprocess.run(command)
    
    # Return the configuration for logging
    return args_dict

def run_learning_rate_experiments(base_args):
    """
    Run experiments with different learning rates
    """
    print("\n=== RUNNING LEARNING RATE EXPERIMENTS ===")
    
    # Define learning rates to test
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
    
    experiments = []
    for lr in learning_rates:
        args = base_args.copy()
        args["learning_rate"] = lr
        args["output_dir"] = f"./logs_lr_{lr}"
        experiments.append(args)
    
    # Run experiments
    for exp in experiments:
        run_experiment(exp)
    
    # Compile results
    compile_results("learning_rate", [f"./logs_lr_{lr}" for lr in learning_rates])

def run_batch_size_experiments(base_args):
    """
    Run experiments with different batch sizes
    """
    print("\n=== RUNNING BATCH SIZE EXPERIMENTS ===")
    
    # Define batch sizes to test
    batch_sizes = [64, 128, 256, 512]
    
    experiments = []
    for bs in batch_sizes:
        args = base_args.copy()
        args["batch_size"] = bs
        args["output_dir"] = f"./logs_bs_{bs}"
        experiments.append(args)
    
    # Run experiments
    for exp in experiments:
        run_experiment(exp)
    
    # Compile results
    compile_results("batch_size", [f"./logs_bs_{bs}" for bs in batch_sizes])

def run_chunk_size_experiments(base_args):
    """
    Run experiments with different episode lengths (chunk sizes)
    """
    print("\n=== RUNNING CHUNK SIZE EXPERIMENTS ===")
    
    # Define chunk sizes to test
    chunk_sizes = [50, 100, 200, 400]
    
    experiments = []
    for cs in chunk_sizes:
        args = base_args.copy()
        args["chunk_size"] = cs
        args["output_dir"] = f"./logs_chunk_{cs}"
        experiments.append(args)
    
    # Run experiments
    for exp in experiments:
        run_experiment(exp)
    
    # Compile results
    compile_results("chunk_size", [f"./logs_chunk_{cs}" for cs in chunk_sizes])

def run_reward_function_experiments(base_args):
    """
    Run experiments with different reward functions
    """
    print("\n=== RUNNING REWARD FUNCTION EXPERIMENTS ===")
    
    # Define reward functions to test (0=abs, 1=squared, 2=sqrt, 3=with action penalty)
    reward_functions = [0, 1, 2, 3]
    reward_names = ["abs", "squared", "sqrt", "action_penalty"]
    
    experiments = []
    for rf, name in zip(reward_functions, reward_names):
        args = base_args.copy()
        args["reward"] = rf
        args["output_dir"] = f"./logs_reward_{name}"
        experiments.append(args)
    
    # Run experiments
    for exp in experiments:
        run_experiment(exp)
    
    # Compile results
    compile_results("reward_function", [f"./logs_reward_{name}" for name in reward_names])

def run_algorithm_experiments(base_args):
    """
    Run experiments with different RL algorithms
    """
    print("\n=== RUNNING ALGORITHM EXPERIMENTS ===")
    
    # Define algorithms to test
    algorithms = ["SAC", "PPO", "TD3", "DDPG"]
    
    experiments = []
    for alg in algorithms:
        args = base_args.copy()
        args["model"] = alg
        args["output_dir"] = f"./logs_alg_{alg}"
        experiments.append(args)
    
    # Run experiments
    for exp in experiments:
        run_experiment(exp)
    
    # Compile results
    compile_results("algorithm", [f"./logs_alg_{alg}" for alg in algorithms])

def run_network_arch_experiments(base_args):
    """
    Run experiments with different network architectures
    """
    print("\n=== RUNNING NETWORK ARCHITECTURE EXPERIMENTS ===")
    
    # Define network architectures to test
    architectures = ["64,64", "128,128", "256,256", "512,512"]
    arch_names = ["small", "medium", "large", "xlarge"]
    
    experiments = []
    for arch, name in zip(architectures, arch_names):
        args = base_args.copy()
        args["net_arch"] = arch
        args["output_dir"] = f"./logs_arch_{name}"
        experiments.append(args)
    
    # Run experiments
    for exp in experiments:
        run_experiment(exp)
    
    # Compile results
    compile_results("network_architecture", [f"./logs_arch_{name}" for name in arch_names])

def run_gamma_experiments(base_args):
    """
    Run experiments with different discount factors (gamma)
    """
    print("\n=== RUNNING GAMMA EXPERIMENTS ===")
    
    # Define gamma values to test
    gamma_values = [0.9, 0.95, 0.99, 0.999]
    
    experiments = []
    for gamma in gamma_values:
        args = base_args.copy()
        args["gamma"] = gamma
        args["output_dir"] = f"./logs_gamma_{gamma}"
        experiments.append(args)
    
    # Run experiments
    for exp in experiments:
        run_experiment(exp)
    
    # Compile results
    compile_results("gamma", [f"./logs_gamma_{gamma}" for gamma in gamma_values])

def run_ent_coef_experiments(base_args):
    """
    Run experiments with different entropy coefficients (ent_coef)
    """
    print("\n=== RUNNING ENTROPY COEFFICIENT EXPERIMENTS ===")
    
    # Define entropy coefficient values to test
    ent_coef_values = [-1.0, 0.01, 0.05, 0.1]
    ent_coef_names = ["auto", "0.01", "0.05", "0.1"]
    
    experiments = []
    for ec, name in zip(ent_coef_values, ent_coef_names):
        args = base_args.copy()
        args["ent_coef"] = ec
        args["output_dir"] = f"./logs_ent_coef_{name}"
        experiments.append(args)
    
    # Run experiments
    for exp in experiments:
        run_experiment(exp)
    
    # Compile results
    compile_results("entropy_coefficient", [f"./logs_ent_coef_{name}" for name in ent_coef_names])

def compile_results(experiment_name, log_dirs):
    """
    Compile results from multiple experiments into comparative visualizations
    
    Args:
        experiment_name (str): Name of the experiment parameter being varied
        log_dirs (list): List of directories containing experiment results
    """
    print(f"Compiling results for {experiment_name} experiments...")
    
    # Create directory for comparative results
    results_dir = f"./comparative_results_{experiment_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Collect metrics from each experiment
    metrics_data = []
    params = []
    
    for log_dir in log_dirs:
        # Extract parameter from directory name
        param = log_dir.split('_')[-1]
        params.append(param)
        
        # Load metrics file (find the first metrics file in the directory)
        metrics_files = [f for f in os.listdir(log_dir) if (f.endswith('metrics_chunk100.csv') or f.startswith('sac_metrics_chunk'))]
        if not metrics_files:
            print(f"No metrics found in {log_dir}")
            continue
            
        metrics_file = os.path.join(log_dir, metrics_files[0])
        metrics_df = pd.read_csv(metrics_file)
        
        # Convert to dict and add to data
        metrics_dict = dict(zip(metrics_df['metric'], metrics_df['value']))
        metrics_dict['param'] = param
        metrics_data.append(metrics_dict)
    
    if not metrics_data:
        print("No metrics data found to compile")
        return
    
    # Create comparative plots
    metrics_of_interest = ['MAE', 'MSE', 'RMSE', '95th_Percentile']
    
    # Bar plot comparing key metrics
    plt.figure(figsize=(12, 8))
    metrics_df = pd.DataFrame(metrics_data)
    
    for i, metric in enumerate(metrics_of_interest):
        plt.subplot(2, 2, i+1)
        bars = plt.bar(metrics_df['param'], metrics_df[metric])
        plt.title(f"{metric} by {experiment_name}")
        plt.ylabel(metric)
        plt.xlabel(experiment_name)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.4f}',
                ha='center', 
                va='bottom',
                rotation=0,
                fontsize=8
            )
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"comparison_metrics_{experiment_name}.png"))
    
    # Create summary table
    summary_df = pd.DataFrame(metrics_data)
    summary_file = os.path.join(results_dir, f"summary_{experiment_name}.csv")
    summary_df.to_csv(summary_file, index=False)
    
    print(f"Results compiled and saved to {results_dir}")
    
    # Also load and plot some speed profiles for visual comparison
    plt.figure(figsize=(12, 10))
    
    for i, log_dir in enumerate(log_dirs[:4]):  # Limit to first 4 to avoid overcrowding
        # Find results CSV
        results_files = [f for f in os.listdir(log_dir) if 'test_results' in f]
        if not results_files:
            continue
            
        results_file = os.path.join(log_dir, results_files[0])
        results_df = pd.read_csv(results_file)
        
        param = log_dir.split('_')[-1]
        
        # Plot speed profile (only show first 500 steps for clarity)
        plt.subplot(4, 1, i+1)
        plt.plot(results_df['timestep'][:500], results_df['reference_speed'][:500], 
                 label="Reference Speed", linestyle="--")
        plt.plot(results_df['timestep'][:500], results_df['predicted_speed'][:500], 
                 label=f"Predicted ({param})")
        plt.ylabel("Speed (m/s)")
        plt.title(f"{experiment_name}={param}")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"speed_profiles_{experiment_name}.png"))
    plt.close()
    
def main():
    parser = argparse.ArgumentParser(description="Run RL experiments for speed following")
    parser.add_argument("--experiment", type=str, default="all",
                      choices=["all", "learning_rate", "batch_size", "chunk_size", 
                               "reward", "algorithm", "network_arch", "gamma", "ent_coef"],
                      help="Type of experiment to run")
    parser.add_argument("--timesteps", type=int, default=100000,
                      help="Total timesteps for training")
    
    args = parser.parse_args()
    
    # Base configuration used for all experiments
    base_args = {
        "model": "SAC",
        "learning_rate": 3e-4,
        "batch_size": 256,
        "buffer_size": 200000,
        "chunk_size": 100,
        "gamma": 0.99,
        "tau": 0.005,
        "ent_coef": -1.0,  # 'auto'
        "net_arch": "256,256",
        "reward": 0,
        "total_timesteps": args.timesteps
    }
    
    # Create results directory
    os.makedirs("./comparative_results", exist_ok=True)
    
    start_time = time.time()
    
    # Run selected experiment
    if args.experiment == "learning_rate" or args.experiment == "all":
        run_learning_rate_experiments(base_args)
        
    if args.experiment == "batch_size" or args.experiment == "all":
        run_batch_size_experiments(base_args)
        
    if args.experiment == "chunk_size" or args.experiment == "all":
        run_chunk_size_experiments(base_args)
        
    if args.experiment == "reward" or args.experiment == "all":
        run_reward_function_experiments(base_args)
        
    if args.experiment == "algorithm" or args.experiment == "all":
        run_algorithm_experiments(base_args)
        
    if args.experiment == "network_arch" or args.experiment == "all":
        run_network_arch_experiments(base_args)
        
    if args.experiment == "gamma" or args.experiment == "all":
        run_gamma_experiments(base_args)
        
    if args.experiment == "ent_coef" or args.experiment == "all":
        run_ent_coef_experiments(base_args)
    
    end_time = time.time()
    print(f"\nAll experiments completed in {end_time - start_time:.2f} seconds")
    
    # Generate final compilation of all results if all experiments were run
    if args.experiment == "all":
        print("\n=== GENERATING FINAL REPORT ===")
        compile_final_report()
        
def compile_final_report():
    """
    Create a final report compiling the best results from each experiment
    """
    # Create results directory
    final_dir = "./final_report"
    os.makedirs(final_dir, exist_ok=True)
    
    # Find all comparative results directories
    comparative_dirs = [d for d in os.listdir("./") if d.startswith("comparative_results_")]
    
    # Extract best configuration for each experiment type
    best_configs = {}
    
    for dir_name in comparative_dirs:
        param_type = dir_name.replace("comparative_results_", "")
        summary_file = os.path.join(dir_name, f"summary_{param_type}.csv")
        
        if not os.path.exists(summary_file):
            continue
            
        # Load summary data
        summary_df = pd.read_csv(summary_file)
        
        # Find best configuration (lowest MAE)
        if 'MAE' in summary_df.columns and 'param' in summary_df.columns:
            best_idx = summary_df['MAE'].idxmin()
            best_param = summary_df.loc[best_idx, 'param']
            best_mae = summary_df.loc[best_idx, 'MAE']
            best_configs[param_type] = (best_param, best_mae)
    
    # Create summary table of best configurations
    if best_configs:
        best_df = pd.DataFrame([
            {"Parameter": k, "Best Value": v[0], "MAE": v[1]} 
            for k, v in best_configs.items()
        ])
        
        best_df.to_csv(os.path.join(final_dir, "best_configurations.csv"), index=False)
        
        # Create visualization of best configurations
        plt.figure(figsize=(10, 6))
        plt.bar(best_df['Parameter'], best_df['MAE'])
        plt.title("Best MAE by Parameter Type")
        plt.ylabel("Mean Absolute Error")
        plt.xlabel("Parameter Type")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(final_dir, "best_configurations.png"))
        plt.close()
    
    # Copy the best speed profile plots from each experiment
    for dir_name in comparative_dirs:
        param_type = dir_name.replace("comparative_results_", "")
        speed_plot = os.path.join(dir_name, f"speed_profiles_{param_type}.png")
        
        if os.path.exists(speed_plot):
            # Copy to final report directory
            import shutil
            shutil.copy(speed_plot, os.path.join(final_dir, f"best_{param_type}_speed_profile.png"))
    
    print(f"Final report compiled in {final_dir}")

def run_best_configuration_experiment():
    """
    Run an experiment with the best configuration found from previous experiments
    """
    # Load best configurations
    best_config_file = "./final_report/best_configurations.csv"
    
    if not os.path.exists(best_config_file):
        print("No best configuration file found. Run all experiments first.")
        return
        
    best_df = pd.read_csv(best_config_file)
    
    # Build best configuration
    base_args = {
        "model": "SAC",
        "learning_rate": 3e-4,
        "batch_size": 256,
        "buffer_size": 200000,
        "chunk_size": 100,
        "gamma": 0.99,
        "tau": 0.005,
        "ent_coef": -1.0,  # 'auto'
        "net_arch": "256,256",
        "reward": 0,
        "total_timesteps": 100000
    }
    
    # Update with best configurations
    for _, row in best_df.iterrows():
        param = row['Parameter']
        value = row['Best Value']
        
        # Handle special cases for parsing
        if param == "learning_rate":
            try:
                base_args[param] = float(value)
            except:
                pass
        elif param == "network_architecture":
            if value == "small":
                base_args["net_arch"] = "64,64"
            elif value == "medium":
                base_args["net_arch"] = "128,128"
            elif value == "large":
                base_args["net_arch"] = "256,256"
            elif value == "xlarge":
                base_args["net_arch"] = "512,512"
        elif param == "reward_function":
            if value == "abs":
                base_args["reward"] = 0
            elif value == "squared":
                base_args["reward"] = 1
            elif value == "sqrt":
                base_args["reward"] = 2
            elif value == "action_penalty":
                base_args["reward"] = 3
        elif param == "entropy_coefficient":
            if value == "auto":
                base_args["ent_coef"] = -1.0
            else:
                base_args["ent_coef"] = float(value)
        else:
            # Try to convert to appropriate type
            try:
                # Try as int
                base_args[param] = int(value)
            except:
                try:
                    # Try as float
                    base_args[param] = float(value)
                except:
                    # Keep as string
                    base_args[param] = value
    
    # Run the experiment with best configuration
    base_args["output_dir"] = "./best_model"
    run_experiment(base_args)
    
    print("Best configuration experiment completed.")

if __name__ == "__main__":
    main()