import numpy as np
from typing import Dict, List, Optional
from collections import deque


def compute_metrics(training_results: Dict, optimal_path_length: int = -1) -> Dict:
    """Compute summary metrics from training results.

    Args:
        training_results: dict from train_agent()
        optimal_path_length: known optimal path length (-1 if unknown)

    Returns:
        dict with computed metrics
    """
    rewards = np.array(training_results['rewards'])
    steps = np.array(training_results['steps'])
    success = np.array(training_results['success'], dtype=float)

    # Overall metrics
    total_episodes = len(rewards)

    # Final performance (last 100 episodes)
    n = min(100, total_episodes)
    final_rewards = rewards[-n:]
    final_steps = steps[-n:]
    final_success = success[-n:]

    # Rolling success rate (window of 50)
    window = min(50, total_episodes)
    rolling_success = np.convolve(success, np.ones(window)/window, mode='valid')

    # Find convergence episode (first episode where rolling success > 90%)
    convergence_ep = total_episodes
    for i, rs in enumerate(rolling_success):
        if rs >= 0.9:
            convergence_ep = i + window
            break

    # Steps improvement: compare first 50 vs last 50 successful episodes
    early_success_steps = []
    late_success_steps = []
    for i in range(total_episodes):
        if success[i]:
            if i < 50:
                early_success_steps.append(steps[i])
            if i >= total_episodes - 50:
                late_success_steps.append(steps[i])

    metrics = {
        'total_episodes': total_episodes,
        'overall_success_rate': float(np.mean(success)),
        'final_success_rate': float(np.mean(final_success)),
        'final_avg_reward': float(np.mean(final_rewards)),
        'final_avg_steps': float(np.mean(final_steps)),
        'final_std_steps': float(np.std(final_steps)),
        'best_reward': float(np.max(rewards)),
        'best_steps': int(np.min(steps[success.astype(bool)])) if np.any(success) else -1,
        'convergence_episode': convergence_ep,
        'training_time': training_results.get('training_time', 0.0),
        'best_path_steps': training_results.get('best_path_steps', -1),
        'rolling_success': rolling_success.tolist(),
        'optimal_path_length': optimal_path_length,
    }

    if optimal_path_length > 0:
        metrics['optimality_ratio'] = metrics['best_path_steps'] / optimal_path_length
    else:
        metrics['optimality_ratio'] = -1.0

    return metrics


def compute_multi_trial_metrics(all_results: List[Dict], optimal_path_length: int = -1) -> Dict:
    """Aggregate metrics across multiple trials."""
    per_trial = [compute_metrics(r, optimal_path_length) for r in all_results]

    return {
        'num_trials': len(all_results),
        'avg_final_success_rate': np.mean([m['final_success_rate'] for m in per_trial]),
        'std_final_success_rate': np.std([m['final_success_rate'] for m in per_trial]),
        'avg_convergence_episode': np.mean([m['convergence_episode'] for m in per_trial]),
        'std_convergence_episode': np.std([m['convergence_episode'] for m in per_trial]),
        'avg_best_steps': np.mean([m['best_steps'] for m in per_trial if m['best_steps'] > 0]),
        'avg_training_time': np.mean([m['training_time'] for m in per_trial]),
        'per_trial': per_trial,
    }


def format_metrics_table(metrics_ql: Dict, metrics_dq: Dict) -> str:
    """Format comparison metrics as a readable table."""
    header = f"{'Metric':<30} {'Q-Learning':>15} {'Dyna-Q':>15} {'Winner':>10}"
    sep = '-' * 72
    lines = [header, sep]

    comparisons = [
        ('Final Success Rate', 'final_success_rate', 'higher'),
        ('Best Path Steps', 'best_path_steps', 'lower'),
        ('Convergence Episode', 'convergence_episode', 'lower'),
        ('Final Avg Reward', 'final_avg_reward', 'higher'),
        ('Final Avg Steps', 'final_avg_steps', 'lower'),
        ('Training Time (s)', 'training_time', 'lower'),
    ]

    for name, key, better in comparisons:
        v1 = metrics_ql.get(key, 'N/A')
        v2 = metrics_dq.get(key, 'N/A')

        if isinstance(v1, float):
            s1, s2 = f"{v1:.4f}", f"{v2:.4f}"
        else:
            s1, s2 = str(v1), str(v2)

        if better == 'higher':
            winner = 'Q-Learn' if v1 > v2 else 'Dyna-Q' if v2 > v1 else 'Tie'
        else:
            winner = 'Q-Learn' if v1 < v2 else 'Dyna-Q' if v2 < v1 else 'Tie'

        lines.append(f"{name:<30} {s1:>15} {s2:>15} {winner:>10}")

    return '\n'.join(lines)
