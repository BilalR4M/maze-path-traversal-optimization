import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class BaseAgent(ABC):
    """Base class for tabular RL agents with epsilon-greedy exploration."""

    def __init__(self, num_states: int, num_actions: int, alpha: float = 0.1,
                 gamma: float = 0.95, epsilon: float = 1.0,
                 epsilon_min: float = 0.1, epsilon_decay: float = 0.995):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((num_states, num_actions))

    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return int(np.argmax(self.q_table[state]))

    def get_greedy_action(self, state: int) -> int:
        """Greedy action (no exploration)."""
        return int(np.argmax(self.q_table[state]))

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset(self):
        """Reset agent state (Q-table, model, etc)."""
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.epsilon = 1.0

    @abstractmethod
    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool):
        """Update Q-values from experience."""
        pass

    def get_policy(self) -> np.ndarray:
        """Return greedy policy as array of best actions per state."""
        return np.argmax(self.q_table, axis=1)

    def get_state_values(self) -> np.ndarray:
        """Return V(s) = max_a Q(s,a) for each state."""
        return np.max(self.q_table, axis=1)
