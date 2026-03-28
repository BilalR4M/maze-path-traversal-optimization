import numpy as np
from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Off-policy Q-Learning agent.

    Update rule:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    next_action parameter is accepted but ignored (off-policy).
    """

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int = 0, done: bool = False):
        current_q = self.q_table[state, action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (target - current_q)

    def get_name(self) -> str:
        return "Q-Learning"
