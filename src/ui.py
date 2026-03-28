import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import GridWorld
from src.agents.q_learning import QLearningAgent
from src.agents.dyna_q import DynaQAgent
from src.agents.sarsa import SarsaAgent
from src.trainer import train_agent, extract_greedy_path
from src.metrics import compute_metrics

ALGORITHM_NAMES = ["Q-Learning", "Dyna-Q", "SARSA"]

st.set_page_config(page_title="TurtleBot RL Pathfinding", layout="wide")
st.title("TurtleBot Path Traversal Optimization")
st.markdown("**RL Agent Comparison** --- Side-by-side path discovery on a random grid world")

def create_env(size, density, seed):
    return GridWorld(size=size, obstacle_density=density, seed=seed)

def create_agent(name, num_states, num_actions, alpha, gamma, epsilon_decay, n_planning):
    if name == "Q-Learning":
        return QLearningAgent(num_states, num_actions, alpha=alpha, gamma=gamma, epsilon_decay=epsilon_decay)
    elif name == "Dyna-Q":
        return DynaQAgent(num_states, num_actions, n_planning=n_planning, alpha=alpha, gamma=gamma, epsilon_decay=epsilon_decay)
    elif name == "SARSA":
        return SarsaAgent(num_states, num_actions, alpha=alpha, gamma=gamma, epsilon_decay=epsilon_decay)
    return QLearningAgent(num_states, num_actions, alpha=alpha, gamma=gamma, epsilon_decay=epsilon_decay)

def render_grid(env, path=None, title=""):
    size = env.size
    fig, ax = plt.subplots(figsize=(6, 6))
    display = np.zeros((size, size, 3))
    for r in range(size):
        for c in range(size):
            if env.grid[r, c] == 1: display[r, c] = [0.2, 0.2, 0.2]
            elif env.grid[r, c] == 2: display[r, c] = [0.2, 0.8, 0.2]
            elif env.grid[r, c] == 3: display[r, c] = [0.2, 0.4, 1.0]
            else: display[r, c] = [1.0, 1.0, 1.0]
    if path and len(path) > 1:
        for i, (r, c) in enumerate(path):
            if env.grid[r, c] == 0:
                t = i / max(len(path) - 1, 1)
                display[r, c] = [1.0 - 0.6*t, 1.0 - 0.3*t, 0.8 - 0.8*t]
    ax.imshow(display, interpolation="nearest")
    for i in range(size + 1):
        ax.axhline(i - 0.5, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(i - 0.5, color="gray", linewidth=0.5, alpha=0.5)
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            ax.annotate("", xy=(path[i+1][1], path[i+1][0]), xytext=(path[i][1], path[i][0]),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax.plot(env.start[1], env.start[0], "g*", markersize=15, label="Start")
    ax.plot(env.goal[1], env.goal[0], "b*", markersize=15, label="Goal")
    if path and len(path) > 0:
        ax.plot(path[-1][1], path[-1][0], "ro", markersize=10, label="Agent")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.legend(loc="upper right", fontsize=8)
    return fig

def render_q_heatmap(q_table, env, title="Q-Value Heatmap"):
    size = env.size
    state_values = np.max(q_table, axis=1).reshape(size, size)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(state_values, cmap="YlOrRd", interpolation="nearest")
    for r in range(size):
        for c in range(size):
            if env.grid[r, c] == 1:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color="black"))
    ax.plot(env.start[1], env.start[0], "g*", markersize=15)
    ax.plot(env.goal[1], env.goal[0], "b*", markersize=15)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticklabels([]); ax.set_yticklabels([])
    return fig

def render_policy_arrows(q_table, env, title="Learned Policy", color="blue"):
    size = env.size
    fig, ax = plt.subplots(figsize=(6, 6))
    display = np.ones((size, size, 3))
    for r in range(size):
        for c in range(size):
            if env.grid[r, c] == 1: display[r, c] = [0.3, 0.3, 0.3]
    ax.imshow(display, interpolation="nearest")
    arrow_map = {0: (0, -0.4), 1: (0, 0.4), 2: (-0.4, 0), 3: (0.4, 0)}
    for s in range(size * size):
        pos = env.state_to_pos(s)
        if env.grid[pos] == 1: continue
        best_action = np.argmax(q_table[s])
        dx, dy = arrow_map[best_action]
        ax.arrow(pos[1], pos[0], dx, dy, head_width=0.15, head_length=0.1, fc=color, ec=color, alpha=0.7)
    for i in range(size + 1):
        ax.axhline(i - 0.5, color="gray", linewidth=0.3, alpha=0.3)
        ax.axvline(i - 0.5, color="gray", linewidth=0.3, alpha=0.3)
    ax.plot(env.start[1], env.start[0], "g*", markersize=15)
    ax.plot(env.goal[1], env.goal[0], "b*", markersize=15)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticklabels([]); ax.set_yticklabels([])
    return fig

# Sidebar Controls
st.sidebar.header("Algorithm Selection")
algo_a = st.sidebar.selectbox("Algorithm A (Left)", ALGORITHM_NAMES, index=0)
algo_b = st.sidebar.selectbox("Algorithm B (Right)", ALGORITHM_NAMES, index=2)

st.sidebar.header("Environment Settings")
grid_size = st.sidebar.slider("Grid Size", 5, 25, 15, step=1)
obstacle_density = st.sidebar.slider("Obstacle Density", 0.1, 0.4, 0.25, step=0.05)
env_seed = st.sidebar.number_input("Environment Seed", value=42, step=1)

st.sidebar.header("Training Settings")
num_episodes = st.sidebar.slider("Number of Episodes", 100, 2000, 500, step=100)
max_steps = st.sidebar.slider("Max Steps per Episode", 100, 1000, 500, step=100)
alpha = st.sidebar.slider("Learning Rate (alpha)", 0.01, 0.5, 0.1, step=0.01)
gamma = st.sidebar.slider("Discount Factor (gamma)", 0.5, 0.99, 0.95, step=0.01)
epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.98, 0.999, 0.995, step=0.001)
n_planning = st.sidebar.slider("Dyna-Q Planning Steps", 1, 20, 5, step=1)

st.sidebar.header("Visualization")
viz_update_freq = st.sidebar.slider("Update Every N Episodes", 1, 50, 10, step=1)

# Main App
if "env" not in st.session_state:
    st.session_state.env = None
    st.session_state.training_done = False
    st.session_state.agent_a_metrics = None
    st.session_state.agent_b_metrics = None
    st.session_state.agent_a_trained = None
    st.session_state.agent_b_trained = None

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("Generate New Environment", use_container_width=True):
        st.session_state.env = create_env(grid_size, obstacle_density, env_seed)
        st.session_state.training_done = False
        st.session_state.agent_a_metrics = None
        st.session_state.agent_b_metrics = None
        st.session_state.agent_a_trained = None
        st.session_state.agent_b_trained = None
        st.rerun()

with col_btn2:
    if st.button("Random Environment", use_container_width=True):
        random_seed = np.random.randint(0, 100000)
        st.session_state.env = create_env(grid_size, obstacle_density, random_seed)
        st.session_state.training_done = False
        st.session_state.agent_a_metrics = None
        st.session_state.agent_b_metrics = None
        st.session_state.agent_a_trained = None
        st.session_state.agent_b_trained = None
        st.rerun()

if st.session_state.env is None:
    st.session_state.env = create_env(grid_size, obstacle_density, env_seed)

env = st.session_state.env

st.subheader("Environment")
col_env1, col_env2 = st.columns(2)
with col_env1:
    fig = render_grid(env, title=f"Grid World ({grid_size}x{grid_size})")
    st.pyplot(fig)
    plt.close(fig)

with col_env2:
    optimal = env.get_optimal_path_length()
    st.metric("Optimal Path Length", optimal if optimal > 0 else "No path")
    st.metric("Grid Size", f"{env.size} x {env.size}")
    st.metric("Obstacles", f"{int(np.sum(env.grid == 1))}")
    st.metric("Free Cells", f"{int(np.sum(env.grid == 0)) + 2}")
    st.metric("Start", str(env.start))
    st.metric("Goal", str(env.goal))

# Training
if st.button(f"Train: {algo_a} vs {algo_b}", use_container_width=True, type="primary"):
    agent_a = create_agent(algo_a, env.num_states, env.num_actions, alpha, gamma, epsilon_decay, n_planning)
    agent_b = create_agent(algo_b, env.num_states, env.num_actions, alpha, gamma, epsilon_decay, n_planning)

    st.subheader("Training Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()
    col_t1, col_t2 = st.columns(2)
    ph1 = col_t1.empty()
    ph2 = col_t2.empty()
    mph = st.empty()

    np.random.seed(42)
    agent_a.reset()
    agent_b.reset()

    a_rewards, b_rewards = [], []
    a_success, b_success = [], []
    a_best_path, b_best_path = None, None
    a_best_steps, b_best_steps = float('inf'), float('inf')

    start_time = time.time()

    for ep in range(num_episodes):
        # Train Agent A
        state = env.reset()
        action = agent_a.select_action(state)
        total_a, path_a = 0.0, [env.agent_pos]
        for s in range(max_steps):
            next_state, reward, done = env.step(action)
            next_action = agent_a.select_action(next_state) if not done else 0
            agent_a.update(state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action
            total_a += reward
            path_a.append(env.agent_pos)
            if done: break
        agent_a.decay_epsilon()
        success_a = (env.agent_pos == env.goal)
        a_success.append(success_a)
        a_rewards.append(total_a)
        if success_a and len(path_a) - 1 < a_best_steps:
            a_best_steps = len(path_a) - 1
            a_best_path = path_a.copy()

        # Train Agent B
        state = env.reset()
        action = agent_b.select_action(state)
        total_b, path_b = 0.0, [env.agent_pos]
        for s in range(max_steps):
            next_state, reward, done = env.step(action)
            next_action = agent_b.select_action(next_state) if not done else 0
            agent_b.update(state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action
            total_b += reward
            path_b.append(env.agent_pos)
            if done: break
        agent_b.decay_epsilon()
        success_b = (env.agent_pos == env.goal)
        b_success.append(success_b)
        b_rewards.append(total_b)
        if success_b and len(path_b) - 1 < b_best_steps:
            b_best_steps = len(path_b) - 1
            b_best_path = path_b.copy()

        if (ep + 1) % viz_update_freq == 0 or ep == num_episodes - 1:
            progress_bar.progress((ep + 1) / num_episodes)
            status_text.text(f"Episode {ep+1}/{num_episodes}")
            window = min(50, len(a_rewards))
            with ph1:
                fig1 = render_grid(env, path=a_best_path, title=f"{algo_a} (Best: {a_best_steps})")
                st.pyplot(fig1)
                plt.close(fig1)
            with ph2:
                fig2 = render_grid(env, path=b_best_path, title=f"{algo_b} (Best: {b_best_steps})")
                st.pyplot(fig2)
                plt.close(fig2)
            with mph.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(f"{algo_a[:3]} Reward", f"{np.mean(a_rewards[-window:]):.1f}")
                m2.metric(f"{algo_b[:3]} Reward", f"{np.mean(b_rewards[-window:]):.1f}")
                m3.metric(f"{algo_a[:3]} Success", f"{np.mean(a_success[-window:])*100:.1f}%")
                m4.metric(f"{algo_b[:3]} Success", f"{np.mean(b_success[-window:])*100:.1f}%")

    t_time = time.time() - start_time
    st.session_state.agent_a_metrics = {"rewards": a_rewards, "success": a_success, "best_path": a_best_path, "best_path_steps": a_best_steps, "training_time": t_time/2}
    st.session_state.agent_b_metrics = {"rewards": b_rewards, "success": b_success, "best_path": b_best_path, "best_path_steps": b_best_steps, "training_time": t_time/2}
    st.session_state.agent_a_trained = agent_a
    st.session_state.agent_b_trained = agent_b
    st.session_state.training_done = True
    st.rerun()

# Results
if st.session_state.training_done and st.session_state.agent_a_metrics:
    st.divider()
    st.header("Results")
    a_m = st.session_state.agent_a_metrics
    b_m = st.session_state.agent_b_metrics
    agent_a = st.session_state.agent_a_trained
    agent_b = st.session_state.agent_b_trained

    st.subheader("Learned Paths (Greedy Policy)")
    col_p1, col_p2 = st.columns(2)
    a_greedy = extract_greedy_path(env, agent_a)
    b_greedy = extract_greedy_path(env, agent_b)
    with col_p1:
        fig = render_grid(env, path=a_greedy, title=f"{algo_a} Path ({len(a_greedy)-1} steps)")
        st.pyplot(fig)
        plt.close(fig)
    with col_p2:
        fig = render_grid(env, path=b_greedy, title=f"{algo_b} Path ({len(b_greedy)-1} steps)")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Metrics Comparison")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric(f"{algo_a} Best Steps", a_m["best_path_steps"])
    col_m2.metric(f"{algo_b} Best Steps", b_m["best_path_steps"])
    col_m3.metric(f"{algo_a} Success", f"{np.mean(a_m['success'])*100:.1f}%")
    col_m4.metric(f"{algo_b} Success", f"{np.mean(b_m['success'])*100:.1f}%")

    st.subheader("Learning Curves")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    window = 50
    a_s = np.convolve(a_m["rewards"], np.ones(window)/window, mode="valid")
    b_s = np.convolve(b_m["rewards"], np.ones(window)/window, mode="valid")
    axes[0].plot(a_s, label=algo_a, alpha=0.8)
    axes[0].plot(b_s, label=algo_b, alpha=0.8)
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Reward"); axes[0].set_title("Reward (smoothed)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    a_sr = np.convolve(a_m["success"], np.ones(window)/window, mode="valid")
    b_sr = np.convolve(b_m["success"], np.ones(window)/window, mode="valid")
    axes[1].plot(a_sr*100, label=algo_a, alpha=0.8)
    axes[1].plot(b_sr*100, label=algo_b, alpha=0.8)
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Success %"); axes[1].set_title("Success Rate")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(np.cumsum(a_m["rewards"]), label=algo_a, alpha=0.8)
    axes[2].plot(np.cumsum(b_m["rewards"]), label=algo_b, alpha=0.8)
    axes[2].set_xlabel("Episode"); axes[2].set_ylabel("Cumulative"); axes[2].set_title("Cumulative Reward")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Q-Value Heatmaps")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        fig = render_q_heatmap(agent_a.q_table, env, f"{algo_a} Q-Values")
        st.pyplot(fig); plt.close(fig)
    with col_h2:
        fig = render_q_heatmap(agent_b.q_table, env, f"{algo_b} Q-Values")
        st.pyplot(fig); plt.close(fig)

    st.subheader("Learned Policies")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        fig = render_policy_arrows(agent_a.q_table, env, f"{algo_a} Policy", "blue")
        st.pyplot(fig); plt.close(fig)
    with col_a2:
        fig = render_policy_arrows(agent_b.q_table, env, f"{algo_b} Policy", "purple")
        st.pyplot(fig); plt.close(fig)
