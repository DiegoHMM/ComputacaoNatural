import numpy as np
import torch
from genetic import RandomLinearQNet
from game import SnakeGameAI, Direction, Point
from model import QTrainer
from collections import deque
import random

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
BLOCK_SIZE = 20


class Individual():
    def __init__(self, input_size, output_size, hidden_layers_range, hidden_size_range, name):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers_range = random.randint(*hidden_layers_range)
        self.hidden_size_range = random.randint(*hidden_size_range)
        self.fitness = 0
        self.model = RandomLinearQNet(self.input_size, self.output_size, self.hidden_layers_range, self.hidden_size_range)
        self.game = SnakeGameAI()
        #
        self.name = name
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def train(self):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        while self.n_games < 100: #run for 100 games
            state_old = self.get_state(self.game)
            final_move = self.get_action(state_old)
            reward, done, score = self.game.play_step(final_move)
            state_new = self.get_state(self.game)
            # Train short memory
            self.train_short_memory(state_old, final_move, reward, state_new, done)
            # Remember
            self.remember(state_old, final_move, reward, state_new, done)
            if done:
                # Train long memory, plot result
                self.game.reset()
                self.n_games += 1
                self.train_long_memory()

                if score > record:
                    record = score
                    self.save_indv(self.name)

    #load model
    def load_indv(self,file_name):
        path = "./model/" + file_name + ".pth"
        # Load the checkpoint
        model_dict = torch.load(path)
        print(model_dict)
        self.model = RandomLinearQNet(model_dict['input_size'], model_dict['output_size'], model_dict['hidden_layers'], model_dict['hidden_size'])
        self.model.load_state_dict(model_dict['state_dict'])

    def save_indv(self, file_name):
        path = "./model/" + file_name + ".pth"
        torch.save({
            'state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers_range,
            'hidden_size': self.hidden_size_range,
            'output_size': self.output_size,
            'fitness': self.fitness,
        }, path)


    def evaluate(self):
        fitness = 0
        for i in range(10): #evaluate game in 10 games
            #testa o modelo numa instância do game e retorna o score
            self.game.reset()
            done = False
            while not done:
                state_old = self.get_state(self.game)
                final_move = self.get_action_trained(state_old)
                reward, done, score = self.game.play_step(final_move)
            fitness += score
        self.fitness = fitness/10
        return self.fitness



    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        #random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    def get_action_trained(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move = [0,0,0]
        final_move[move] = 1
        return final_move