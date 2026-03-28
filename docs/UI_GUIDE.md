# Using the Streamlit UI - User Guide

## Overview

The UI provides a side-by-side comparison of two reinforcement learning agents navigating a random grid world. You can choose any combination of **Q-Learning**, **Dyna-Q**, and **SARSA** to compare.

## Interface Layout

```
+------------------------------------------------------------------+
|  TurtleBot Path Traversal Optimization                           |
+------------------------------------------------------------------+
| SIDEBAR (Settings)        | MAIN AREA                            |
| - Algorithm Selection     | - Environment Grid                   |
| - Environment Settings    | - Metrics                            |
| - Training Settings       | - Training Progress (during training)|
| - Visualization           | - Results (after training)           |
+---------------------------+--------------------------------------+
```

## Step-by-Step Guide

### Step 1: Select Algorithms

In the **sidebar**, choose which two algorithms to compare:

| Algorithm | Type | Key Property |
|-----------|------|--------------|
| Q-Learning | Off-policy | Uses max Q(s',a) for updates |
| Dyna-Q | Model-based | Q-Learning + planning with learned model |
| SARSA | On-policy | Uses Q(s',a') where a' is actual next action |

**Recommended comparisons:**
- **Q-Learning vs Dyna-Q**: See how planning accelerates learning
- **Q-Learning vs SARSA**: See off-policy vs on-policy behavioral differences

### Step 2: Configure Environment

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Grid Size | 5-25 | 15 | Size of the grid (NxN) |
| Obstacle Density | 0.1-0.4 | 0.25 | Fraction of cells that are obstacles |
| Environment Seed | Any integer | 42 | Controls random grid generation |

Click **"Generate New Environment"** to create a new random grid with the current settings, or **"Random Environment"** for a random seed.

### Step 3: Configure Training

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Number of Episodes | 100-2000 | 500 | Total training episodes |
| Max Steps per Episode | 100-1000 | 500 | Steps before forced termination |
| Learning Rate (α) | 0.01-0.5 | 0.1 | How fast the agent learns |
| Discount Factor (γ) | 0.5-0.99 | 0.95 | Importance of future rewards |
| Epsilon Decay | 0.98-0.999 | 0.995 | How fast exploration decreases |
| Dyna-Q Planning Steps | 1-20 | 5 | Planning updates per real step (Dyna-Q only) |

### Step 4: Train the Agents

Click the **"Train: [Algo A] vs [Algo B]"** button.

During training, you'll see:
- **Progress bar**: Shows training completion percentage
- **Live grids**: Best path found so far for each agent
- **Live metrics**: Rolling average reward and success rate

### Step 5: View Results

After training completes, scroll down to see:

#### Learned Paths (Greedy Policy)
The best path found by each agent, displayed on the grid with arrows.

| Visual Element | Color |
|----------------|-------|
| Free cells | White |
| Obstacles | Dark gray |
| Start | Green star |
| Goal | Blue star |
| Path trail | Orange-red gradient |
| Path arrows | Red |

#### Metrics Comparison
| Metric | What it means |
|--------|---------------|
| Best Path Steps | Shortest path found (lower is better) |
| Success Rate | % of episodes reaching the goal |

#### Learning Curves
Three charts showing:
1. **Reward per Episode** (smoothed) - Higher is better
2. **Rolling Success Rate** - Shows learning progress
3. **Cumulative Reward** - Total reward over time

#### Q-Value Heatmaps
Heatmap of learned state values (brighter = higher expected reward).

#### Learned Policies
Arrow visualization showing the best action from each state.

## Understanding Results

### Which algorithm wins?

| Scenario | Expected Winner | Why |
|----------|-----------------|-----|
| Convergence speed | Dyna-Q | Planning accelerates learning |
| Final path quality | Tie | Both find optimal given enough time |
| Conservative paths | SARSA | Accounts for exploration risk |
| Raw speed | Q-Learning | Less computation per step |

### Key Observations

1. **Dyna-Q learns faster**: With planning steps, it updates Q-values from simulated experiences, needing fewer real environment interactions.

2. **SARSA is more cautious**: Near obstacles, SARSA learns safer policies because it accounts for the exploration noise in its updates.

3. **Q-Learning can be riskier**: Uses max Q-value (assuming optimal behavior) but actual behavior during training includes random exploration.

## Tips

- Start with **Grid Size = 10** for quick testing
- Use **500+ episodes** for reliable results
- Try **different seeds** to see variance
- For SARSA vs Q-Learning comparison, try a grid with obstacles near the edges to see behavioral differences

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Stop Streamlit server |
| `R` | Rerun the app |
| `C` | Clear cache |
