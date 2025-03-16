import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

import torch
import torch.nn as nn
import os
import time
import argparse
import csv

# ------------------------------------------------------------------------
# 1) Always create a 1200-step speed dataset
# ------------------------------------------------------------------------
DATA_LEN = 1200
CSV_FILE = "speed_profile.csv"

# Force-generate a 1200-step sinusoidal + noise speed profile
speeds = 10 + 5 * np.sin(0.02 * np.arange(DATA_LEN)) + 2 * np.random.randn(DATA_LEN)
df_fake = pd.DataFrame({"speed": speeds})
df_fake.to_csv(CSV_FILE, index=False)
print(f"Created {CSV_FILE} with {DATA_LEN} steps.")

df = pd.read_csv(CSV_FILE)
full_speed_data = df["speed"].values
assert len(full_speed_data) == DATA_LEN, "Dataset must be 1200 steps after generation."

# ------------------------------------------------------------------------
# 2) Utility: chunk the dataset, possibly with leftover
# ------------------------------------------------------------------------
def chunk_into_episodes(data, chunk_size):
    """
    Splits `data` into chunks of length `chunk_size`.
    If leftover < chunk_size remains, it becomes a smaller final chunk.
    """
    episodes = []
    start = 0
    while start < len(data):
        end = start + chunk_size
        chunk = data[start:end]
        episodes.append(chunk)
        start = end
    return episodes

# ------------------------------------------------------------------------
# 3A) Training Environment: picks a random chunk each reset
# ------------------------------------------------------------------------
class TrainEnv(gym.Env):
    """
    Speed-following training environment:
      - The dataset is split into episodes of length `chunk_size`.
      - Each reset(), we pick one chunk at random.
      - action: acceleration in [-3,3]
      - observation: [current_speed, reference_speed]
      - reward: -|current_speed - reference_speed|
    """

    def __init__(self, episodes_list, delta_t=1.0, reward_type=0):
        super().__init__()
        self.episodes_list = episodes_list
        self.num_episodes = len(episodes_list)
        self.delta_t = delta_t
        self.reward_type = reward_type

        # Actions, Observations
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=50.0, shape=(2,), dtype=np.float32)

        # Episode-specific
        self.current_episode = None
        self.episode_len = 0
        self.step_idx = 0
        self.current_speed = 0.0
        self.ref_speed = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Pick random chunk from episodes_list
        ep_idx = np.random.randint(0, self.num_episodes)
        self.current_episode = self.episodes_list[ep_idx]
        self.episode_len = len(self.current_episode)
        self.step_idx = 0

        # Initialize
        self.current_speed = 0.0
        self.ref_speed = self.current_episode[self.step_idx]

        obs = np.array([self.current_speed, self.ref_speed], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        accel = np.clip(action[0], -3.0, 3.0)
        self.current_speed += accel * self.delta_t
        if self.current_speed < 0:
            self.current_speed = 0.0

        self.ref_speed = self.current_episode[self.step_idx]
        error = abs(self.current_speed - self.ref_speed)
        
        # Different reward functions based on reward_type
        if self.reward_type == 0:
            reward = -error  # Default: Negative absolute error
        elif self.reward_type == 1:
            reward = -error**2  # Squared error (penalizes larger errors more)
        elif self.reward_type == 2:
            reward = -np.sqrt(error)  # Square root (softer penalty for large errors)
        elif self.reward_type == 3:
            reward = -error - 0.1 * abs(accel)  # Penalize large actions too
        else:
            reward = -error

        self.step_idx += 1
        terminated = (self.step_idx >= self.episode_len)
        truncated = False

        obs = np.array([self.current_speed, self.ref_speed], dtype=np.float32)
        info = {"speed_error": error}
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------------
# 3B) Testing Environment: run entire 1200-step data in one episode
# ------------------------------------------------------------------------
class TestEnv(gym.Env):
    """
    Speed-following testing environment:
      - We run through the entire 1200-step dataset in one go.
      - observation: [current_speed, reference_speed]
      - reward: -|current_speed - reference_speed|
    """

    def __init__(self, full_data, delta_t=1.0, reward_type=0):
        super().__init__()
        self.full_data = full_data
        self.n_steps = len(full_data)
        self.delta_t = delta_t
        self.reward_type = reward_type

        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=50.0, shape=(2,), dtype=np.float32)

        self.idx = 0
        self.current_speed = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.current_speed = 0.0
        ref_speed = self.full_data[self.idx]
        obs = np.array([self.current_speed, ref_speed], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        accel = np.clip(action[0], -3.0, 3.0)
        self.current_speed += accel * self.delta_t
        if self.current_speed < 0:
            self.current_speed = 0.0

        ref_speed = self.full_data[self.idx]
        error = abs(self.current_speed - ref_speed)
        
        # Different reward functions based on reward_type
        if self.reward_type == 0:
            reward = -error  # Default: Negative absolute error
        elif self.reward_type == 1:
            reward = -error**2  # Squared error (penalizes larger errors more)
        elif self.reward_type == 2:
            reward = -np.sqrt(error)  # Square root (softer penalty for large errors)
        elif self.reward_type == 3:
            reward = -error - 0.1 * abs(accel)  # Penalize large actions too
        else:
            reward = -error

        self.idx += 1
        terminated = (self.idx >= self.n_steps)
        truncated = False

        obs = np.array([self.current_speed, ref_speed], dtype=np.float32)
        info = {"speed_error": error}
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------------
# 4) CustomLoggingCallback (optional)
# ------------------------------------------------------------------------
from stable_baselines3.common.callbacks import BaseCallback

class CustomLoggingCallback(BaseCallback):
    def __init__(self, log_dir, log_name="training_log.csv", verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_path = os.path.join(log_dir, log_name)
        self.episode_rewards = []
        os.makedirs(log_dir, exist_ok=True)
        with open(self.log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestep', 'average_reward'])

    def _on_step(self):
        t = self.num_timesteps
        reward = self.locals.get('rewards', [0])[-1]
        self.episode_rewards.append(reward)

        if self.locals.get('dones', [False])[-1]:
            avg_reward = np.mean(self.episode_rewards)
            with open(self.log_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([t, avg_reward])
            self.logger.record("reward/average_reward", avg_reward)
            self.episode_rewards.clear()

        return True


# ------------------------------------------------------------------------
# 5) Main: user sets chunk_size from command line, train, then test
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs_chunk_training",
        help="Directory to store logs and trained model."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Episode length for training (e.g. 50, 100, 200)."
    )
    # Add model selection argument
    parser.add_argument(
        "--model",
        type=str,
        default="SAC",
        choices=["SAC", "PPO", "TD3", "DDPG"],
        help="RL algorithm to use (SAC, PPO, TD3, DDPG)."
    )
    # Add hyperparameter arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training."
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=200000,
        help="Replay buffer size."
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Soft update coefficient."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor."
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=-1.0,
        help="Entropy coefficient. Use -1.0 for 'auto'."
    )
    parser.add_argument(
        "--net_arch",
        type=str,
        default="256,256",
        help="Network architecture (comma-separated list of layer sizes)."
    )
    parser.add_argument(
        "--reward",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Reward function: 0=neg_abs_error, 1=neg_squared_error, 2=neg_sqrt_error, 3=error+action_penalty"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100000,
        help="Total timesteps for training."
    )
    
    args = parser.parse_args()

    log_dir = args.output_dir
    
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["stdout", "tensorboard"])

    chunk_size = args.chunk_size
    print(f"[INFO] Using chunk_size = {chunk_size}")
    print(f"[INFO] Selected model: {args.model}")

    # Parse net_arch from string to list of integers
    net_arch = [int(size) for size in args.net_arch.split(",")]
    print(f"[INFO] Network architecture: {net_arch}")

    # 5A) Split the 1200-step dataset into chunk_size episodes
    episodes_list = chunk_into_episodes(full_speed_data, chunk_size)
    print(f"Number of episodes: {len(episodes_list)} (some leftover if 1200 not divisible by {chunk_size})")

    # 5B) Create the TRAIN environment
    def make_train_env():
        return TrainEnv(episodes_list, delta_t=1.0, reward_type=args.reward)

    train_env = DummyVecEnv([make_train_env])

    # 5C) Build the model based on selected algorithm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    policy_kwargs = dict(net_arch=net_arch, activation_fn=nn.ReLU)
    
    # Handle ent_coef special case
    ent_coef = 'auto' if args.ent_coef == -1.0 else args.ent_coef
    
    # Common parameters for all algorithms
    common_params = {
        "policy": "MlpPolicy",
        "env": train_env,
        "verbose": 1,
        "policy_kwargs": policy_kwargs,
        "learning_rate": args.learning_rate,
        "device": device,
        "gamma": args.gamma,
    }
    
    # Initialize the selected algorithm with appropriate parameters
    if args.model == "SAC":
        model = SAC(
            **common_params,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            tau=args.tau,
            ent_coef=ent_coef
        )
    elif args.model == "PPO":
        model = PPO(
            **common_params,
            batch_size=args.batch_size,
            n_steps=1024,  # PPO-specific parameter
            ent_coef=0.01 if args.ent_coef == -1.0 else args.ent_coef
        )
    elif args.model == "TD3":
        model = TD3(
            **common_params,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            tau=args.tau
        )
    elif args.model == "DDPG":
        model = DDPG(
            **common_params,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            tau=args.tau
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.set_logger(logger)

    total_timesteps = args.total_timesteps
    callback = CustomLoggingCallback(log_dir)

    print(f"[INFO] Start training for {total_timesteps} timesteps...")
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=100,
        callback=callback
    )
    end_time = time.time()
    print(f"[INFO] Training finished in {end_time - start_time:.2f}s")

    # 5D) Save the model
    save_path = os.path.join(log_dir, f"{args.model.lower()}_speed_follow_chunk{chunk_size}")
    model.save(save_path)
    print(f"[INFO] Model saved to: {save_path}.zip")

    # ------------------------------------------------------------------------
    # 5E) Test the model on the FULL 1200-step dataset in one go
    # ------------------------------------------------------------------------
    test_env = TestEnv(full_speed_data, delta_t=1.0, reward_type=args.reward)

    obs, _ = test_env.reset()
    predicted_speeds = []
    reference_speeds = []
    rewards = []
    actions = []  # Track actions for analysis

    for _ in range(DATA_LEN):
        action, _states = model.predict(obs, deterministic=True)
        actions.append(action[0])  # Store the action
        obs, reward, terminated, truncated, info = test_env.step(action)
        predicted_speeds.append(obs[0])  # current_speed
        reference_speeds.append(obs[1])  # reference_speed
        rewards.append(reward)
        if terminated or truncated:
            break

    avg_test_reward = np.mean(rewards)
    avg_speed_error = np.mean([abs(p - r) for p, r in zip(predicted_speeds, reference_speeds)])
    print(f"[TEST] Average reward over 1200-step test: {avg_test_reward:.3f}")
    print(f"[TEST] Average speed error: {avg_speed_error:.3f}")

    # Calculate quantitative metrics
    errors = np.array([abs(p - r) for p, r in zip(predicted_speeds, reference_speeds)])
    squared_errors = errors**2
    
    # Mean metrics
    mae = np.mean(errors)
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    
    # Calculate convergence rate (how quickly error reduces over time)
    # Using exponential moving average to smooth the data
    window_size = 50
    smoothed_errors = np.convolve(errors, np.ones(window_size)/window_size, mode='valid')
    
    # Calculate error reduction rate (negative means error is decreasing)
    if len(smoothed_errors) > 1:
        # Linear regression on log of smoothed errors
        x = np.arange(len(smoothed_errors))
        # Add small constant to avoid log(0)
        y = np.log(smoothed_errors + 1e-10)
        
        # Simple linear regression
        n = len(x)
        slope = (n * np.sum(x*y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        convergence_rate = slope
    else:
        convergence_rate = 0.0
    
    # Percentile metrics
    p95_error = np.percentile(errors, 95)
    p99_error = np.percentile(errors, 99)
    
    print(f"[METRICS] Mean Absolute Error (MAE): {mae:.4f}")
    print(f"[METRICS] Mean Squared Error (MSE): {mse:.4f}")
    print(f"[METRICS] Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"[METRICS] 95th Percentile Error: {p95_error:.4f}")
    print(f"[METRICS] 99th Percentile Error: {p99_error:.4f}")
    print(f"[METRICS] Error Convergence Rate: {convergence_rate:.6f}")

    # Add metrics to test results dataframe
    results_path = os.path.join(log_dir, f"{args.model.lower()}_test_results_chunk{chunk_size}.csv")
    results_df = pd.DataFrame({
        "timestep": range(len(predicted_speeds)),
        "reference_speed": reference_speeds,
        "predicted_speed": predicted_speeds,
        "error": errors,
        "squared_error": squared_errors,
        "reward": rewards,
        "action": actions
    })
    results_df.to_csv(results_path, index=False)
    print(f"[INFO] Test results saved to: {results_path}")

    # Save metrics to separate file
    metrics_path = os.path.join(log_dir, f"{args.model.lower()}_metrics_chunk{chunk_size}.csv")
    metrics_df = pd.DataFrame({
        'metric': ['MAE', 'MSE', 'RMSE', '95th_Percentile', '99th_Percentile', 'Convergence_Rate'],
        'value': [mae, mse, rmse, p95_error, p99_error, convergence_rate]
    })
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[INFO] Metrics saved to: {metrics_path}")

    # Create multiple plots for all metrics and save them
    # 1. Speed tracking plot (reference vs predicted)
    plt.figure(figsize=(10, 6))
    plt.plot(reference_speeds, label="Reference Speed", linestyle="--")
    plt.plot(predicted_speeds, label="Predicted Speed", linestyle="-")
    plt.xlabel("Timestep")
    plt.ylabel("Speed (m/s)")
    plt.title(f"Speed Tracking Performance ({args.model}, chunk_size={chunk_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    speed_plot_path = os.path.join(log_dir, f"1_speed_tracking_plot.png")
    plt.savefig(speed_plot_path)
    plt.close()
    
    # 2. Error plot
    plt.figure(figsize=(10, 6))
    plt.plot(errors, label="Absolute Error", color="red")
    
    # Add moving average for trend visualization
    plt.plot(np.arange(window_size//2, window_size//2 + len(smoothed_errors)), 
             smoothed_errors, label=f"Moving Avg (window={window_size})", 
             color="darkred", linewidth=2)
    
    plt.axhline(y=mae, color='black', linestyle='--', label=f"MAE = {mae:.4f}")
    plt.xlabel("Timestep")
    plt.ylabel("Error")
    plt.title(f"Speed Tracking Error ({args.model}, chunk_size={chunk_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    error_plot_path = os.path.join(log_dir, f"2_error_plot.png")
    plt.savefig(error_plot_path)
    plt.close()
    
    # 3. Squared error plot
    plt.figure(figsize=(10, 6))
    plt.plot(squared_errors, label="Squared Error", color="purple")
    plt.axhline(y=mse, color='black', linestyle='--', label=f"MSE = {mse:.4f}")
    plt.xlabel("Timestep")
    plt.ylabel("Squared Error")
    plt.title(f"Speed Tracking Squared Error ({args.model}, chunk_size={chunk_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    squared_error_plot_path = os.path.join(log_dir, f"3_squared_error_plot.png")
    plt.savefig(squared_error_plot_path)
    plt.close()
    
    # 4. Action plot
    plt.figure(figsize=(10, 6))
    plt.plot(actions, label="Action (Acceleration)", color="green")
    plt.xlabel("Timestep")
    plt.ylabel("Acceleration")
    plt.title(f"Control Actions ({args.model}, chunk_size={chunk_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    action_plot_path = os.path.join(log_dir, f"4_action_plot.png")
    plt.savefig(action_plot_path)
    plt.close()
    
    # 5. Error histogram
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, color="blue")
    plt.axvline(x=mae, color='red', linestyle='--', label=f"MAE = {mae:.4f}")
    plt.axvline(x=rmse, color='green', linestyle='--', label=f"RMSE = {rmse:.4f}")
    plt.axvline(x=p95_error, color='purple', linestyle='--', label=f"95% = {p95_error:.4f}")
    plt.xlabel("Error Value")
    plt.ylabel("Frequency")
    plt.title(f"Error Distribution ({args.model}, chunk_size={chunk_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    hist_plot_path = os.path.join(log_dir, f"5_error_histogram.png")
    plt.savefig(hist_plot_path)
    plt.close()
    
    # 6. Combined visualization plot (keep this one for display)
    plt.figure(figsize=(12, 10))
    
    # Speed tracking
    plt.subplot(3, 1, 1)
    plt.plot(reference_speeds, label="Reference Speed", linestyle="--")
    plt.plot(predicted_speeds, label="Predicted Speed", linestyle="-")
    plt.ylabel("Speed (m/s)")
    plt.title(f"Performance Summary ({args.model}, chunk={chunk_size}, MAE={mae:.3f}, MSE={mse:.3f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Errors
    plt.subplot(3, 1, 2)
    plt.plot(errors, label="Absolute Error", color="red", alpha=0.5)
    plt.plot(np.arange(window_size//2, window_size//2 + len(smoothed_errors)), 
             smoothed_errors, label="Smoothed Error", color="darkred")
    plt.axhline(y=mae, color='black', linestyle='--', label=f"MAE = {mae:.4f}")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Actions
    plt.subplot(3, 1, 3)
    plt.plot(actions, label="Acceleration", color="green")
    plt.xlabel("Timestep")
    plt.ylabel("Action")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    combined_plot_path = os.path.join(log_dir, f"6_combined_plot.png")
    plt.savefig(combined_plot_path)
    plt.close()

    # Print summary of hyperparameters and metrics
    print("\n--- Training Configuration and Results Summary ---")
    print(f"Algorithm: {args.model}")
    print(f"Chunk size: {chunk_size}")
    print(f"Network architecture: {net_arch}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Reward function: {args.reward}")
    if args.model in ["SAC", "TD3", "DDPG"]:
        print(f"Buffer size: {args.buffer_size}")
        print(f"Tau: {args.tau}")
    if args.model == "SAC":
        print(f"Entropy coefficient: {ent_coef}")
    print(f"Gamma: {args.gamma}")
    print(f"Total timesteps: {total_timesteps}")
    print("\n--- Performance Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"95th Percentile Error: {p95_error:.4f}")
    print(f"99th Percentile Error: {p99_error:.4f}")
    print(f"Error Convergence Rate: {convergence_rate:.6f}")
    print("-----------------------------------")


if __name__ == "__main__":
    main()