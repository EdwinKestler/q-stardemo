import numpy as np
import random

class Gridworld:
    def __init__(self, size=5, obstacles=2, goal=(4, 4)):
        self.size = size
        self.obstacles = obstacles
        self.goal = goal
        self.grid = np.zeros((size, size))
        self.set_obstacles()
        self.grid[goal] = 2  # Representing the goal with 2

    def set_obstacles(self):
        count = 0
        while count < self.obstacles:
            x, y = np.random.randint(0, self.size, 2)
            if (x, y) != self.goal and self.grid[x, y] == 0:
                self.grid[x, y] = -1  # Representing obstacles with -1
                count += 1

    def reset(self):
        self.position = (0, 0)
        while self.grid[self.position] == -1:  # Ensure not starting on an obstacle
            self.position = tuple(np.random.randint(0, self.size, 2))
        return self.position

    def step(self, action):
        x, y = self.position
        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Down
            x = min(self.size - 1, x + 1)
        elif action == 2:  # Left
            y = max(0, y - 1)
        elif action == 3:  # Right
            y = min(self.size - 1, y + 1)

        self.position = (x, y)

        if self.position == self.goal:
            return self.position, 10, True  # Goal reached
        elif self.grid[self.position] == -1:
            return self.position, -10, True  # Hit obstacle
        else:
            return self.position, -1, False  # Normal move

    def render(self):
        display_grid = self.grid.copy()
        display_grid[self.position] = 3  # Representing the agent with 3
        print(display_grid)


class QLearningAgent:
    def __init__(self, actions, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.q_table = dict()
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha     # Learning rate
        self.gamma = gamma     # Discount factor
        self.actions = actions

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:  # Explore
            return random.choice(self.actions)
        else:  # Exploit
            q_values = [self.get_q_value(state, a) for a in self.actions]
            max_q = max(q_values)
            actions_with_max_q = [a for a, q in zip(self.actions, q_values) if q == max_q]
            return random.choice(actions_with_max_q)

    def learn(self, state, action, reward, next_state):
        old_q = self.get_q_value(state, action)
        next_max_q = max([self.get_q_value(next_state, a) for a in self.actions])
        new_q = (1 - self.alpha) * old_q + self.alpha * (reward + self.gamma * next_max_q)
        self.q_table[(state, action)] = new_q


def run_episode(agent, environment, max_steps=50):
    state = environment.reset()
    total_reward = 0
    for _ in range(max_steps):
        action = agent.choose_action(state)
        next_state, reward, done = environment.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        if done:
            break
    return total_reward


# Initialize the environment and agent
env = Gridworld()
agent = QLearningAgent(actions=[0, 1, 2, 3])

# Training the agent
episodes = 200
for episode in range(episodes):
    reward = run_episode(agent, env)
    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {reward}")

# Display the final state of the grid
env.render()
