import numpy as np
from typing import Tuple, Optional, List
from collections import deque


class GridWorld:
    """Grid world environment for RL pathfinding.

    Actions: 0=Up, 1=Down, 2=Left, 3=Right
    Cells: 0=free, 1=obstacle, 2=start, 3=goal
    """

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    ACTIONS = [0, 1, 2, 3]
    ACTION_NAMES = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
    ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    REWARD_STEP = -1
    REWARD_GOAL = 100
    REWARD_OBSTACLE = -100

    def __init__(self, size: int = 15, obstacle_density: float = 0.25, seed: Optional[int] = None):
        self.size = size
        self.obstacle_density = obstacle_density
        self.rng = np.random.RandomState(seed)
        self.grid: Optional[np.ndarray] = None
        self.start: Optional[Tuple[int, int]] = None
        self.goal: Optional[Tuple[int, int]] = None
        self.agent_pos: Optional[Tuple[int, int]] = None
        self.num_states = size * size
        self.num_actions = 4
        self.generate_environment()

    def generate_environment(self):
        """Generate a random grid with obstacles, start, and goal. Guarantees a path exists."""
        self.grid = np.zeros((self.size, self.size), dtype=int)

        num_obstacles = int(self.size * self.size * self.obstacle_density)
        free_cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        self.rng.shuffle(free_cells)

        for i in range(num_obstacles):
            self.grid[free_cells[i]] = 1

        remaining = free_cells[num_obstacles:]
        self.start = remaining[0]
        self.goal = remaining[-1]
        self.grid[self.start] = 2
        self.grid[self.goal] = 3

        if not self._path_exists():
            self.generate_environment()

        self.agent_pos = self.start

    def _path_exists(self) -> bool:
        """BFS check for path from start to goal."""
        visited = {self.start}
        queue = deque([self.start])
        while queue:
            r, c = queue.popleft()
            if (r, c) == self.goal:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size \
                        and (nr, nc) not in visited and self.grid[nr, nc] != 1:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    def pos_to_state(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.size + pos[1]

    def state_to_pos(self, state: int) -> Tuple[int, int]:
        return (state // self.size, state % self.size)

    def reset(self) -> int:
        """Reset agent to start. Returns initial state."""
        self.agent_pos = self.start
        return self.pos_to_state(self.start)

    def step(self, action: int) -> Tuple[int, float, bool]:
        """Execute action. Returns (next_state, reward, done)."""
        dr, dc = self.ACTION_DELTAS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        if not (0 <= nr < self.size and 0 <= nc < self.size) or self.grid[nr, nc] == 1:
            return self.pos_to_state(self.agent_pos), self.REWARD_OBSTACLE, True

        self.agent_pos = (nr, nc)
        if self.agent_pos == self.goal:
            return self.pos_to_state(self.agent_pos), self.REWARD_GOAL, True

        return self.pos_to_state(self.agent_pos), self.REWARD_STEP, False

    def is_free(self, r: int, c: int) -> bool:
        if 0 <= r < self.size and 0 <= c < self.size:
            return self.grid[r, c] != 1
        return False

    def get_optimal_path_length(self) -> int:
        """BFS shortest path length from start to goal."""
        visited = {self.start: 0}
        queue = deque([(self.start, 0)])
        while queue:
            (r, c), dist = queue.popleft()
            if (r, c) == self.goal:
                return dist
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size \
                        and (nr, nc) not in visited and self.grid[nr, nc] != 1:
                    visited[(nr, nc)] = dist + 1
                    queue.append(((nr, nc), dist + 1))
        return -1

    def get_neighbors(self, state: int) -> List[Tuple[int, int, int]]:
        """Return list of (next_state, reward, done) for all valid actions."""
        results = []
        pos = self.state_to_pos(state)
        for action in self.ACTIONS:
            dr, dc = self.ACTION_DELTAS[action]
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < self.size and 0 <= nc < self.size) or self.grid[nr, nc] == 1:
                results.append((state, self.REWARD_OBSTACLE, True))
            elif (nr, nc) == self.goal:
                results.append((self.pos_to_state((nr, nc)), self.REWARD_GOAL, True))
            else:
                results.append((self.pos_to_state((nr, nc)), self.REWARD_STEP, False))
        return results
