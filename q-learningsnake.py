import pygame
import random
import numpy as np
import math


# Initialize Pygame
pygame.init()

# ... [Color definitions, display settings, etc., same as before] ...
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)
 
dis_width = 600
dis_height = 400
 
dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('Q learning Snake Game by EdwinK')
 
clock = pygame.time.Clock()
 
snake_block = 10
snake_speed = 15
 
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)

def calculate_angle(snake_head, food_position):
    return math.atan2(food_position[1] - snake_head[1], food_position[0] - snake_head[0])

def calculate_distance(snake_head, food_position):
    return math.sqrt((food_position[0] - snake_head[0]) ** 2 + (food_position[1] - snake_head[1]) ** 2)

def is_danger_close(snake_head, snake_list, direction):
    x, y = snake_head
    danger_front, danger_left, danger_right = False, False, False

    # Check danger in front
    if direction == "UP":
        danger_front = (x, y - snake_block) in snake_list or y - snake_block < 0
        danger_left = (x - snake_block, y) in snake_list or x - snake_block < 0
        danger_right = (x + snake_block, y) in snake_list or x + snake_block >= dis_width
    elif direction == "DOWN":
        danger_front = (x, y + snake_block) in snake_list or y + snake_block >= dis_height
        danger_left = (x + snake_block, y) in snake_list or x + snake_block >= dis_width
        danger_right = (x - snake_block, y) in snake_list or x - snake_block < 0
    elif direction == "LEFT":
        danger_front = (x - snake_block, y) in snake_list or x - snake_block < 0
        danger_left = (x, y + snake_block) in snake_list or y + snake_block >= dis_height
        danger_right = (x, y - snake_block) in snake_list or y - snake_block < 0
    elif direction == "RIGHT":
        danger_front = (x + snake_block, y) in snake_list or x + snake_block >= dis_width
        danger_left = (x, y - snake_block) in snake_list or y - snake_block < 0
        danger_right = (x, y + snake_block) in snake_list or y + snake_block >= dis_height

    return (danger_front, danger_left, danger_right)
 
# QLearningAgent class definition
class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.q_table = dict()  # Initialize Q-table
        self.alpha = alpha     # Learning rate
        self.gamma = gamma     # Discount factor
        self.epsilon = epsilon # Exploration rate
        self.actions = actions
        self.epsilon_decay = epsilon_decay

    # Method to get Q-value
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    # Method to choose action
    def choose_action(self, state):
        if random.random() < self.epsilon:  # Explore
            return random.choice(self.actions)
        else:  # Exploit
            q_values = [self.get_q_value(state, a) for a in self.actions]
            max_q = max(q_values)
            actions_with_max_q = [a for a, q in zip(self.actions, q_values) if q == max_q]
            return random.choice(actions_with_max_q)

    # Method to learn and update Q-table
    def learn(self, state, action, reward, next_state):
        old_q = self.get_q_value(state, action)
        next_max_q = max([self.get_q_value(next_state, a) for a in self.actions])
        new_q = (1 - self.alpha) * old_q + self.alpha * (reward + self.gamma * next_max_q)
        self.q_table[(state, action)] = new_q
        self.epsilon *= self.epsilon_decay

# ... [Methods for score, snake drawing, messages, remain the same] ...

def Your_score(score):
    value = score_font.render("Your Score: " + str(score), True, yellow)
    dis.blit(value, [0, 0])
 
  
def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(dis, black, [x[0], x[1], snake_block, snake_block])
 
 
def message(msg, color):
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [dis_width / 6, dis_height / 3])

# Game loop with Q-learning
def gameLoop():
    # Speed levels
    speed_levels = {pygame.K_1: 10, pygame.K_2: 20, pygame.K_3: 30}
    global snake_speed
    
    # Initialize the QLearningAgent
    agent = QLearningAgent(actions=["LEFT", "STRAIGHT", "RIGHT"])

    
    while True:  # Main game loop
        # Initialize the direction variable
        direction = "UP"
        game_over = False
        game_close = False
        # Game initialization variables
        x1, y1 = dis_width / 2, dis_height / 2
        x1_change, y1_change = 0, 0
        snake_List = []
        Length_of_snake = 1
        foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
        foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0

        while not game_over:
            while game_close:
                # Display losing message
                dis.fill(blue)
                message("You Lost! Press Q-Quit", red)
                Your_score(Length_of_snake - 1)
                pygame.display.update()
                
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            pygame.quit()
                            quit()
                        else:
                            game_close = False

            # Define the current state with more details
            angle_to_food = calculate_angle((x1, y1), (foodx, foody))
            distance_to_food = calculate_distance((x1, y1), (foodx, foody))
            danger = is_danger_close((x1, y1), snake_List, direction)
            # Define the current state
            state = (x1, y1, direction, angle_to_food, distance_to_food, danger)

            # Let the agent choose the action
            action = agent.choose_action(state)

            # Apply the chosen action to update direction
            if action == "LEFT":
                if direction == "UP":
                    direction = "LEFT"
                elif direction == "DOWN":
                    direction = "RIGHT"
                elif direction == "LEFT":
                    direction = "DOWN"
                elif direction == "RIGHT":
                    direction = "UP"
            elif action == "STRAIGHT":
                # No change in direction
                pass
            elif action == "RIGHT":
                if direction == "UP":
                    direction = "RIGHT"
                elif direction == "DOWN":
                    direction = "LEFT"
                elif direction == "LEFT":
                    direction = "UP"
                elif direction == "RIGHT":
                    direction = "DOWN"

            # Update position of the snake based on the direction
            if direction == "UP":
                y1_change = -snake_block
                x1_change = 0
            elif direction == "DOWN":
                y1_change = snake_block
                x1_change = 0
            elif direction == "LEFT":
                x1_change = -snake_block
                y1_change = 0
            elif direction == "RIGHT":
                x1_change = snake_block
                y1_change = 0

            # Apply the position change
            x1 += x1_change
            y1 += y1_change

            # Check for collisions with walls or itself
            if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
                game_close = True
            for segment in snake_List[:-1]:
                if segment == [x1, y1]:
                    game_close = True

            # Update the state after the action
            next_state = (x1, y1, direction)

            # Calculate the reward
            reward = calculate_reward(game_close, x1, y1, foodx, foody)

            # Let the agent learn from the experience
            agent.learn(state, action, reward, next_state)

            # Render the snake and food
            dis.fill(blue)
            pygame.draw.rect(dis, green, [foodx, foody, snake_block, snake_block])
            our_snake(snake_block, snake_List)
            Your_score(Length_of_snake - 1)
            pygame.display.update()

            # Check if the snake has eaten the food
            if x1 == foodx and y1 == foody:
                foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
                foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
                Length_of_snake += 1

            # Update snake's body
            snake_Head = []
            snake_Head.append(x1)
            snake_Head.append(y1)
            snake_List.append(snake_Head)
            if len(snake_List) > Length_of_snake:
                del snake_List[0]

            clock.tick(snake_speed)
            
            # Restart the game if closed
            if game_close:
                game_close = False
                break
                
    pygame.quit()
    quit()

def calculate_reward(game_close, x1, y1, foodx, foody):
    if game_close:
        return -10  # Negative reward for losing
    elif x1 == foodx and y1 == foody:
        return 10  # Positive reward for eating food
    else:
        return -1  # Slight negative reward for each step to encourage efficiency

gameLoop()