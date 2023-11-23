import numpy as np
import random

class EnhancedGridworld:
    def __init__(self, size=5, moving_obstacles=2, static_obstacles=1, goals=[(4, 4)], goal_rewards=[10]):
        self.size = size
        self.grid = np.zeros((size, size))
        self.goals = goals
        self.goal_rewards = goal_rewards
        self.set_static_obstacles(static_obstacles)
        self.set_moving_obstacles(moving_obstacles)

    def set_static_obstacles(self, count):
        while count > 0:
            x, y = np.random.randint(0, self.size, 2)
            if (x, y) not in self.goals and self.grid[x, y] == 0:
                self.grid[x, y] = -1  # Static obstacles
                count -= 1

    def set_moving_obstacles(self, count):
        self.moving_obstacles = []
        while count > 0:
            x, y = np.random.randint(0, self.size, 2)
            if (x, y) not in self.goals and self.grid[x, y] == 0:
                self.moving_obstacles.append((x, y))
                count -= 1

    def reset(self):
        self.position = (0, 0)
        while self.grid[self.position] == -1 or self.position in self.moving_obstacles:
            self.position = tuple(np.random.randint(0, self.size, 2))
        return self.position

    def step(self, action):
        # Agent movement logic
        x, y = self.position
        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Down
            x = min(self.size - 1, x + 1)
        elif action == 2:  # Left
            y = max(0, y - 1)
        elif action == 3:  # Right
            y = min(self.size - 1, y + 1)

        new_position = (x, y)

        # Check for collisions with static or moving obstacles
        if self.grid[new_position] == -1 or new_position in self.moving_obstacles:
            return new_position, -10, True  # Collision

        # Check for reaching a goal
        for goal, reward in zip(self.goals, self.goal_rewards):
            if new_position == goal:
                return new_position, reward, True  # Goal reached

        # Update agent position
        self.position = new_position

        # Move dynamic obstacles
        self.move_dynamic_obstacles()

        return new_position, -1, False  # Normal step

    def move_dynamic_obstacles(self):
        for i, obs in enumerate(self.moving_obstacles):
            x, y = obs
            move = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])  # Random move
            new_x, new_y = x + move[0], y + move[1]
            new_x = max(0, min(new_x, self.size - 1))
            new_y = max(0, min(new_y, self.size - 1))
            self.moving_obstacles[i] = (new_x, new_y)

    def render(self):
        display_grid = self.grid.copy()
        for pos in self.moving_obstacles:
            display_grid[pos] = -2  # Moving obstacles
        for goal in self.goals:
            display_grid[goal] = 2  # Goals
        display_grid[self.position] = 3  # Agent
        print(display_grid)


# QLearningAgent class remains the same as in the previous example
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



# Initialize the enhanced environment and agent
enhanced_env = EnhancedGridworld(goals=[(3, 3), (4, 4)], goal_rewards=[5, 10], moving_obstacles=2, static_obstacles=1)
enhanced_agent = QLearningAgent(actions=[0, 1, 2, 3])

# Training the enhanced agent
episodes = 200
for episode in range(episodes):
    reward = run_episode(enhanced_agent, enhanced_env)
    if episode % 50 == 0:
        print(f"Enhanced Episode {episode}, Reward: {reward}")

# Display the final state of the enhanced grid
enhanced_env.render()
