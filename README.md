# Reinforcement Learning for Speed Profile Following

This repository contains code for training and evaluating reinforcement learning agents to follow speed profiles. The project compares different RL algorithms and hyperparameters to find the optimal configuration for a speed-following control task.

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/alexneilgreen/Reinforcement-Learning-Exercise.git
cd Reinforcement-Learning-Exercise
```

### Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- numpy
- pandas
- matplotlib
- gymnasium
- stable-baselines3
- tensorboard
- torch

## Usage

### Running Experiments

The main script `Main.py` can be used to run a variety of experiments with different hyperparameters. The script manages training, testing, and evaluation of models automatically.

#### Basic Usage

To run all experiments with default settings:

```bash
python Main.py
```

#### Specific Experiments

You can run specific experiments by specifying the experiment type and number of timesteps:

```bash
# Run only learning rate experiments
python Main.py --experiment learning_rate --timesteps 100000

# Run only batch size experiments
python Main.py --experiment batch_size --timesteps 50000
```

#### Available Experiment Types

- `learning_rate`: Tests different learning rates
- `batch_size`: Tests different batch sizes
- `chunk_size`: Tests different episode lengths
- `reward`: Tests different reward functions
- `algorithm`: Tests different RL algorithms (SAC, PPO, TD3, DDPG)
- `network_arch`: Tests different neural network architectures
- `gamma`: Tests different discount factors
- `ent_coef`: Tests different entropy coefficients

#### Command Line Arguments

- `--experiment`: Type of experiment to run (`all`, `learning_rate`, `batch_size`, etc.)
- `--timesteps`: Total timesteps for training (default: 100000)

### Understanding Results

After running experiments, results are stored in the following directories:

- Individual experiment results: `./logs_*` directories
- Comparative results: `./comparative_results_*` directories
- Final report (when running all experiments): `./final_report`

Each experiment directory contains:

- Trained model files (`.zip`)
- Test results (`.csv`)
- Performance metrics (`.csv`)
- Visualization plots (`.png`)

## Project Structure

- `Main.py`: Main script for running experiments and comparing results
- `RL_assignment.py`: Core implementation of RL environments and training logic
- `speed_profile.csv`: Generated speed profile data for training and testing

## Example Commands

```bash
# Run all experiments with 200,000 timesteps
python Main.py --experiment all --timesteps 200000

# Run only algorithm comparison with 100,000 timesteps
python Main.py --experiment algorithm --timesteps 100000

# Run only reward function experiments (Default Timesteps 100,000)
python Main.py --experiment reward
```

## Visualizations

The experiments generate several visualizations to help understand model performance:

1. Speed tracking plot: Shows how well the agent follows the reference speed
2. Error plot: Shows absolute error over time
3. Squared error plot: Shows squared error over time
4. Action plot: Shows control actions taken by the agent
5. Error histogram: Shows the distribution of errors
6. Combined visualization: Combines speed tracking, errors, and actions in one plot

Additional comparative visualizations can be found in the associated `./comparative_results_*` directories.

The final report directory (`./final_report`) also has a few final summary visualizations.

## Notes

- The code first generates a synthetic speed profile with 1200 steps
- For training, the data is split into chunks of specified size
- For testing, the agent runs through the entire 1200-step profile
- Performance is measured using MAE, MSE, RMSE, and percentile metrics
