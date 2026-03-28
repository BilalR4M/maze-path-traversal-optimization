# Running Jupyter Notebooks

## Prerequisites

Make sure you have the virtual environment set up and dependencies installed:

```bash
# From the project root
source venv/Scripts/activate  # Git Bash
# or
venv\Scripts\activate         # CMD/PowerShell

# If not already installed
pip install -r requirements.txt
pip install jupyter ipykernel
```

## Starting Jupyter

```bash
# From the project root directory
jupyter notebook
```

This opens Jupyter in your browser at `http://localhost:8888`. Navigate to the `notebooks/` folder.

## Available Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_q_learning.ipynb` | Train and analyze Q-Learning agent |
| 2 | `02_dyna_q.ipynb` | Train and analyze Dyna-Q agent |
| 3 | `03_comparison.ipynb` | Q-Learning vs Dyna-Q comparison |
| 4 | `04_sarsa.ipynb` | Train and analyze SARSA agent |
| 5 | `05_qlearning_vs_sarsa.ipynb` | Q-Learning vs SARSA comparison |

## Running a Notebook

1. Open the notebook from Jupyter's file browser
2. Click on the first cell
3. Press `Shift+Enter` to run each cell, or use `Cell > Run All` to run all cells

## Notebook Contents

### 01_q_learning.ipynb
- Sets up a 15x15 grid world
- Trains Q-Learning agent for 500 episodes
- Shows learning curves (reward, success rate, steps, epsilon)
- Displays Q-value heatmap and learned policy arrows
- Extracts and visualizes the greedy path
- Dynamic animation of path discovery over training

### 02_dyna_q.ipynb
- Same as above but for Dyna-Q agent
- Also shows model growth (number of learned transitions)
- Demonstrates faster convergence due to planning

### 03_comparison.ipynb
- Trains both Q-Learning and Dyna-Q on the same environment
- Side-by-side learning curves (6 plots)
- Q-value heatmaps comparison
- Policy arrows comparison
- Greedy paths comparison
- Statistical metrics comparison
- Multi-trial analysis (10 trials) with box plots

### 04_sarsa.ipynb
- Trains SARSA (on-policy) agent
- Shows learning curves, heatmaps, policies
- Demonstrates conservative behavior near obstacles

### 05_qlearning_vs_sarsa.ipynb
- Compares off-policy (Q-Learning) vs on-policy (SARSA)
- Highlights behavioral differences near obstacles
- Statistical comparison with learning curves
- Multi-trial analysis

## Configuration

Each notebook has configuration cells at the top. You can modify:

```python
GRID_SIZE = 15          # Grid dimensions (15x15)
OBSTACLE_DENSITY = 0.25 # 25% of cells are obstacles
SEED = 42               # Random seed for reproducibility
NUM_EPISODES = 500      # Training episodes
MAX_STEPS = 500         # Max steps per episode
ALPHA = 0.1             # Learning rate
GAMMA = 0.95            # Discount factor
EPSILON_DECAY = 0.995   # Exploration decay rate
N_PLANNING = 5          # Dyna-Q planning steps per real step
```

## Expected Output

Each notebook will:
1. Print environment details (grid size, optimal path length)
2. Show training progress (best path, training time)
3. Display matplotlib plots inline

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: src` | Run from project root or add `sys.path.insert` |
| Plots not showing | Use `%matplotlib inline` in first cell |
| Kernel not found | Run `python -m ipykernel install --user` |
| Slow training | Reduce `NUM_EPISODES` or `GRID_SIZE`
