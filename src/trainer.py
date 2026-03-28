import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from .environment import GridWorld
from .agents.base_agent import BaseAgent


def train_agent(env: GridWorld, agent: BaseAgent, num_episodes: int = 500,
                max_steps: int = 500, seed: Optional[int] = None) -> Dict:
    """Train agent for num_episodes. Returns metrics dict.

    Works for Q-Learning, Dyna-Q, and SARSA.
    SARSA uses next_action, Q-Learning/Dyna-Q ignore it.
    """
    if seed is not None:
        np.random.seed(seed)

    agent.reset()

    metrics = {
        'rewards': [],
        'steps': [],
        'success': [],
        'epsilons': [],
        'best_path': None,
        'best_path_steps': float('inf'),
        'training_time': 0.0,
        'convergence_episode': num_episodes,
        'q_table_snapshots': [],
    }

    start_time = time.time()

    for ep in range(num_episodes):
        state = env.reset()
        # Pre-select first action (needed for SARSA)
        action = agent.select_action(state)
        total_reward = 0.0
        path = [env.agent_pos]

        for step in range(max_steps):
            next_state, reward, done = env.step(action)
            
            # Select next action BEFORE update (SARSA needs it)
            if done:
                next_action = 0  # dummy value, not used when done=True
            else:
                next_action = agent.select_action(next_state)

            # Pass next_action to update
            # Q-Learning/Dyna-Q accept **kwargs or ignore extra params
            # SARSA uses it for on-policy update
            agent.update(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action  # current becomes next for next iteration
            total_reward += reward
            path.append(env.agent_pos)

            if done:
                break

        agent.decay_epsilon()

        success = (env.agent_pos == env.goal)
        metrics['rewards'].append(total_reward)
        metrics['steps'].append(len(path) - 1)
        metrics['success'].append(success)
        metrics['epsilons'].append(agent.epsilon)

        if success and len(path) - 1 < metrics['best_path_steps']:
            metrics['best_path_steps'] = len(path) - 1
            metrics['best_path'] = path.copy()

        if ep >= 49:
            rolling_success = np.mean(metrics['success'][-50:])
            if rolling_success >= 0.9 and metrics['convergence_episode'] == num_episodes:
                metrics['convergence_episode'] = ep

        if ep % 50 == 0 or ep == num_episodes - 1:
            metrics['q_table_snapshots'].append({
                'episode': ep, 'q_table': agent.q_table.copy()
            })

    metrics['training_time'] = time.time() - start_time
    return metrics


def extract_greedy_path(env: GridWorld, agent: BaseAgent,
                        max_steps: int = 500) -> List[Tuple[int, int]]:
    """Extract greedy path from learned Q-table (epsilon=0)."""
    state = env.reset()
    path = [env.agent_pos]
    visited = set()
    visited.add(env.agent_pos)

    for _ in range(max_steps):
        action = agent.get_greedy_action(state)
        next_state, reward, done = env.step(action)
        state = next_state

        if env.agent_pos in visited and not done:
            break
        visited.add(env.agent_pos)
        path.append(env.agent_pos)

        if done:
            break

    return path


def run_single_trial(env: GridWorld, agent: BaseAgent, num_episodes: int = 500,
                     max_steps: int = 500, seed: int = 0) -> Dict:
    """Run a single training trial with a given seed."""
    return train_agent(env, agent, num_episodes, max_steps, seed=seed)


def run_multiple_trials(env: GridWorld, agent_factory, num_trials: int = 10,
                        num_episodes: int = 500, max_steps: int = 500,
                        base_seed: int = 42) -> List[Dict]:
    """Run multiple independent trials. agent_factory creates fresh agents."""
    results = []
    for i in range(num_trials):
        agent = agent_factory()
        metrics = run_single_trial(env, agent, num_episodes, max_steps,
                                   seed=base_seed + i)
        results.append(metrics)
    return results
