import numpy as np
from .base_agent import BaseAgent


class SarsaAgent(BaseAgent):
    """On-policy SARSA agent.

    Update rule:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
    where a' is the action actually taken (epsilon-greedy), not greedy max.
    """

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int, done: bool):
        current_q = self.q_table[state, action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state, next_action]
        self.q_table[state, action] += self.alpha * (target - current_q)

    def get_name(self) -> str:
        return "SARSA"
