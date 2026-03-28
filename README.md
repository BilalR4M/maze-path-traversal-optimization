# TurtleBot Path Traversal Optimization

Reinforcement learning agents learning to navigate a 2D grid world to find optimal paths from start to goal while avoiding obstacles.

## Algorithms Implemented

| Algorithm | Type | Description |
|-----------|------|-------------|
| **Q-Learning** | Off-policy, model-free | Classic TD control using max Q(s',a) |
| **Dyna-Q** | Off-policy, model-based | Q-Learning + planning with learned model |
| **SARSA** | On-policy, model-free | TD control using actual next action Q(s',a') |

## Project Structure

```
turtlebot-path-traversal-optimization/
├── src/
│   ├── __init__.py
│   ├── environment.py       # GridWorld MDP environment
│   ├── trainer.py           # Unified training loop
│   ├── metrics.py           # Performance metrics computation
│   ├── ui.py                # Streamlit web UI
│   └── agents/
│       ├── __init__.py
│       ├── base_agent.py    # Abstract base class
│       ├── q_learning.py    # Q-Learning agent
│       ├── dyna_q.py        # Dyna-Q agent
│       └── sarsa.py         # SARSA agent
├── notebooks/
│   ├── 01_q_learning.ipynb
│   ├── 02_dyna_q.ipynb
│   ├── 03_comparison.ipynb  # Q-Learning vs Dyna-Q
│   ├── 04_sarsa.ipynb
│   └── 05_qlearning_vs_sarsa.ipynb
├── docs/
│   ├── RUN_NOTEBOOKS.md     # Guide for running notebooks
│   ├── RUN_UI.md            # Guide for running Streamlit UI
│   └── UI_GUIDE.md          # User guide for the UI
├── requirements.txt
└── venv/                    # Virtual environment
```

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/Scripts/activate  # Git Bash
# or
venv\Scripts\activate         # CMD/PowerShell

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Streamlit UI

```bash
streamlit run src/ui.py
```

Opens at `http://localhost:8501`. See [docs/UI_GUIDE.md](docs/UI_GUIDE.md) for usage.

### 3. Run Jupyter Notebooks

```bash
jupyter notebook
```

Navigate to `notebooks/` and open any notebook. See [docs/RUN_NOTEBOOKS.md](docs/RUN_NOTEBOOKS.md) for details.

## Features

### Streamlit UI
- Side-by-side comparison of any two algorithms
- Random environment generation
- Real-time training visualization
- Learning curves, Q-value heatmaps, policy arrows
- Configurable hyperparameters

### Jupyter Notebooks
- Individual agent analysis (01, 02, 04)
- Algorithm comparisons (03, 05)
- Dynamic path discovery animations
- Multi-trial statistical analysis
- Comprehensive visualizations

## Environment

- **Grid World**: 2D grid with configurable size (default 15x15)
- **Actions**: Up, Down, Left, Right
- **Rewards**: -1 per step, +100 for goal, -100 for obstacle collision
- **Obstacles**: Randomly placed (default 25% density)
- **Guarantee**: BFS ensures a path always exists

## Key Dependencies

- `numpy` - Numerical arrays
- `streamlit` - Web UI
- `matplotlib` - Plotting
- `pandas` - Data analysis (optional)

## References

- Sutton & Barto, "Reinforcement Learning: An Introduction"
- Q-Learning: Off-policy TD control
- Dyna-Q: Integration of learning and planning
- SARSA: On-policy TD control
