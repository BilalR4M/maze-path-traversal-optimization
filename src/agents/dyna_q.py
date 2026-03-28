import numpy as np
from .base_agent import BaseAgent


class DynaQAgent(BaseAgent):
    """Dyna-Q agent: Q-Learning + planning with a learned environment model.

    After each real step, performs n_planning simulated updates using
    randomly sampled experiences from the model.
    next_action parameter is accepted but ignored (off-policy).
    """

    def __init__(self, num_states: int, num_actions: int,
                 n_planning: int = 5, **kwargs):
        super().__init__(num_states, num_actions, **kwargs)
        self.n_planning = n_planning
        self.model = {}  # (state, action) -> (next_state, reward)

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int = 0, done: bool = False):
        # Real experience Q-update (off-policy, ignores next_action)
        current_q = self.q_table[state, action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (target - current_q)

        # Store transition in model
        self.model[(state, action)] = (next_state, reward)

        # Planning: sample from model and do extra Q-updates
        if len(self.model) > 0:
            keys = list(self.model.keys())
            for _ in range(min(self.n_planning, len(keys))):
                idx = np.random.randint(len(keys))
                s_p, a_p = keys[idx]
                ns_p, r_p = self.model[(s_p, a_p)]
                q_p = self.q_table[s_p, a_p]
                target_p = r_p + self.gamma * np.max(self.q_table[ns_p])
                self.q_table[s_p, a_p] += self.alpha * (target_p - q_p)

    def reset(self):
        """Reset Q-table and model."""
        super().reset()
        self.model = {}

    def get_name(self) -> str:
        return "Dyna-Q"
