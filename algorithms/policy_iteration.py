import numpy as np
import gymnasium as gym
from grid_world_env.envs.grid_world import GridWorldEnv, Actions


class PolicyIteration:
    def __init__(self, env, theta, gamma) -> None:
        self.env = env
        self.size = env.size

        self.theta = theta
        self.gamma = gamma

        self.P, self.R, self.terminal, self.cells, self.states_idx = self._build_mdp()

        n_actions = self.env.action_space.n  # type: ignore

        self.pi = np.ones((self.n_states, n_actions)) / n_actions
        self.V = np.zeros(self.n_states)

    def reset(self):
        self.env.reset()

        self.P, self.R, self.terminal, self.cells, self.states_idx = self._build_mdp()

        n_actions = self.env.action_space.n  # type: ignore

        self.pi = np.ones((self.n_states, n_actions)) / n_actions
        self.V = np.zeros(self.n_states)

    def _is_out_of_boundary(self, x, y, size):
        return x < 0 or x >= size or y < 0 or y >= size

    def _build_mdp(self):

        target_location = tuple(self.env._target_location)
        forbidden_locations = self.env._forbidden_locations

        cells = [(x, y) for x in range(self.size) for y in range(self.size)]
        states_idx = {cell: i for i, cell in enumerate(cells)}

        self.n_states = len(cells)
        n_actions = self.env.action_space.n  # type: ignore

        action_to_dir = self.env._action_to_direction

        P = np.zeros((self.n_states, n_actions, self.n_states))
        R = np.zeros((self.n_states, n_actions))

        terminal = np.zeros(self.n_states, dtype=bool)
        terminal[states_idx[target_location]] = True

        for x, y in cells:
            for action in range(n_actions):
                dir = action_to_dir[action]
                nx, ny = x + dir[0], y + dir[1]

                if (
                    self._is_out_of_boundary(nx, ny, self.size)
                    or (nx, ny) in forbidden_locations
                ):
                    nx, ny = nx - dir[0], ny - dir[1]
                    reward = -1
                elif (nx, ny) == target_location:
                    reward = 1
                else:
                    reward = 0

                state_id = states_idx[(x, y)]
                n_state_id = states_idx[(nx, ny)]
                P[state_id, action, n_state_id] = 1.0
                R[state_id, action] = reward

        return P, R, terminal, cells, states_idx

    def _policy_evaluation(self):
        delta = 0.0
        while True:
            for s in range(self.n_states):
                if self.terminal[s]:
                    continue
                a = self.pi[s]
                v = self.V[s]

                self.V[s] = self.R[s, a] + self.gamma * float(
                    np.dot(self.P[s, a], self.V)
                )
                delta = max(delta, abs(v - self.V[s]))

            if delta < self.theta:
                break


if __name__ == "__main__":

    env = GridWorldEnv()
    env.reset()

    pi = PolicyIteration(env, theta=0.1, gamma=0.5)

    pi._policy_evaluation()

    print(pi.V)

    env.close()
    pass
