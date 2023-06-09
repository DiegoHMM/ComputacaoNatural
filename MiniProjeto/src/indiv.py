import pickle
from collections import deque
from dinamic import *
import random
import copy
import numpy as np
import torch
from grammar import *
from game import SnakeGameAI, Direction, Point
from model import QTrainer, Linear_QNet
from helper import plot
from dinamic import *
#
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
BLOCK_SIZE = 20


class Individual():
    def __init__(self, input_size, n_hidden_layers, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden_layers = n_hidden_layers+1
        self.fitness = 0
        self.grammar = generate_hidden_layers(input_size, self.n_hidden_layers, output_size)
        self.model = DynamicLinear_QNet(self.grammar)
        self.game = SnakeGameAI()
        #
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def set_grammar(self, grammar):
        self.grammar = grammar
    def build_model(self):
        self.model = DynamicLinear_QNet(self.grammar)

    def get_last_actions(self):
        last_actions = []
        if len(self.memory) > 15:
            for i in range(15):
                last_actions.append(self.memory[-i][1])
        return last_actions

    def detect_cycles(self):
        cycle1 = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
        cycle2 = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

        actions = self.get_last_actions()
        if len(actions) >= 15:
            cycle_length = 3  # define the length of the cycle you want to search for
            max_cycle = 5
            detected_cycle1 = 0
            detected_cycle2 = 0
            for i in range(len(actions) - cycle_length+1):
                segment = actions[i:i + cycle_length]  # get a slice of the actions list
                if segment == cycle1:
                    detected_cycle1 += 1
                elif segment == cycle2:
                    detected_cycle2 += 1
                if detected_cycle1 >= max_cycle or detected_cycle2 >= max_cycle:
                    return True
        return False
    def train(self):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0

        while self.n_games < 150:#run for 150 games
            state_old = self.get_state(self.game)
            final_move = self.get_action(state_old)
            reward, done, score = self.game.play_step(final_move)
            state_new = self.get_state(self.game)
            #b_cycle = self.detect_cycles()
            #if b_cycle:
            #    print("Cycle detected")
            #    done = True
            #    reward = -10
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

                #print('Game', self.n_games, 'Score', score, 'Record', record)


                plot_scores.append(score)
                total_score += score
                mean_score = total_score / (self.n_games+1)
                plot_mean_scores.append(mean_score)
                #plot(plot_scores, plot_mean_scores)


    def load_model_structure(self, folder, file_name):
        path = "./" + folder + "/" + file_name + "_structure.pkl"
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        return self.model

    def load_indv(self, folder, file_name):
        path = "./"+folder+"/" + file_name + ".pth"
        # Load the checkpoint
        checkpoint = torch.load(path)
        self.grammar = checkpoint['grammar']
        self.model = self.load_model_structure(folder, file_name) #DynamicLinear_QNet(self.grammar)
        self.model.load_state_dict(checkpoint['state_dict'])

    def save_indv(self, folder, file_name):
        path = "./"+folder+"/" + file_name + ".pth"
        print("Saving model: ", self.grammar)
        torch.save({
            'state_dict': self.model.state_dict(),
            'grammar': self.grammar,
            'fitness': self.fitness,
        }, path)

        self.save_model_structure(folder, file_name)

    def save_model_structure(self, folder, file_name):
        path = "./" + folder + "/" + file_name + "_structure.pkl"
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

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

def tournament_selection(population, tournament_size=2):
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda x: x.fitness, reverse=True)
    best_indv = tournament[0]
    return best_indv

def point_crossover(individual_1, individual_2, max_hidden_layers=1, max_tries=100):
    child_1 = Individual(individual_1.input_size, individual_1.n_hidden_layers, individual_1.output_size)
    child_2 = Individual(individual_2.input_size, individual_2.n_hidden_layers, individual_2.output_size)
    child_1.grammar = individual_1.grammar.copy()
    child_2.grammar = individual_2.grammar.copy()
    #Se existe pelo menos uma camada oculta
    if len(individual_1.grammar[1:-1]) != 0 and len(individual_2.grammar[1:-1]) != 0:
        for i in range(max_tries):
            #select random layer in individual_1 and get the index of the layer
            layer_1 = random.choice(individual_1.grammar[1:-1])
            index_1 = individual_1.grammar.index(layer_1)
            #select random layer in individual_2
            layer_2 = random.choice(individual_2.grammar[1:-1])
            index_2 = individual_2.grammar.index(layer_2)

            if get_activation_function(layer_1) != get_activation_function(layer_2):
                #altere the layers dimensions
                aux_in, aux_out = get_layer_dims(layer_1)
                layer_2_in, layer_2_out = get_layer_dims(layer_2)
                layer_1 = change_layer_dimensions(layer_1, layer_2_in, layer_2_out)
                layer_2 = change_layer_dimensions(layer_2, aux_in, aux_out)
                #swap the layers
                child_1.grammar = individual_1.grammar.copy()
                #put layer_2 in the index of layer_1
                child_1.grammar[index_1] = layer_2
                child_2.grammar = individual_2.grammar.copy()
                #put layer_1 in the index of layer_2
                child_2.grammar[index_2] = layer_1

                return child_1, child_2
            return individual_1, individual_2
    else:
        #50% de chance de trocar a primeira camada dos individuos
        if random.random() < 0.5:
            #select first layer of individual_1 and get the index of the layer
            child_1.grammar[0] = change_activation_function(individual_1.grammar[0],get_activation_function(individual_2.grammar[0]))
            child_2.grammar[0] = change_activation_function(individual_2.grammar[0],get_activation_function(individual_1.grammar[0]))
            return child_1, child_2
        else:
            #select last layer of individual_1 and get the index of the layer
            child_1.grammar[-1] = change_activation_function(individual_1.grammar[-1],get_activation_function(individual_2.grammar[-1]))
            child_2.grammar[-1] = change_activation_function(individual_2.grammar[-1],get_activation_function(individual_1.grammar[-1]))
            return child_1, child_2




def delete_layer(indv_grammar, random_layer):
    prev_layer = indv_grammar[indv_grammar.index(random_layer) - 1]
    next_layer = indv_grammar[indv_grammar.index(random_layer) + 1]
    # get dimensions of random_layer
    in_dim, out_dim = get_layer_dims(random_layer)
    # change dimensions of prev layer
    new_prev_layer = change_layer_dimensions(prev_layer, get_layer_dims(prev_layer)[0], in_dim)
    # change dimensions of next layer
    new_next_layer = change_layer_dimensions(next_layer, in_dim, get_layer_dims(next_layer)[1])
    # change layer of indv_grammar
    indv_grammar[indv_grammar.index(random_layer) - 1] = new_prev_layer
    indv_grammar[indv_grammar.index(random_layer) + 1] = new_next_layer
    indv_grammar.remove(random_layer)
    return indv_grammar


def replace_n_layers(individual_grammar, random_layer):
    in_dim, out_dim = get_layer_dims(random_layer)
    max_layers = verify_length(individual_grammar)
    if max_layers > 0:
        amount_new_layers = random.randint(1, max_layers)
        new_layers = generate_hidden_layers(in_dim, amount_new_layers, out_dim)
        #inserir new_layers elements on place of random_layer
        index = individual_grammar.index(random_layer)
        individual_grammar.pop(index)
        for layer in reversed(new_layers):
            individual_grammar.insert(index, layer)
        return individual_grammar
def clone_layer(individual_grammar, random_layer):
    random_index_layer = individual_grammar.index(random_layer)
    _, out_dim = get_layer_dims(random_layer)
    new_layer = change_layer_dimensions(random_layer, out_dim, out_dim)
    individual_grammar.insert(random_index_layer + 1, new_layer)
    return individual_grammar

#Only inidivuous without hidden layers
def add_random_layer(individual_grammar):
    _ , in_dim = get_layer_dims(individual_grammar[0])
    out_dim, _ = get_layer_dims(individual_grammar[-1])
    new_layer = generate_hidden_layers(in_dim, 0, out_dim)
    individual_grammar.insert(1, new_layer[0])
    return individual_grammar

def mutate(individual): #With elitism
    new_individual = Individual(individual.input_size, individual.n_hidden_layers, individual.output_size)
    individual_grammar = individual.grammar.copy()
    if len(individual_grammar) == 2: #Apenas adiciona uma layer
        new_grammar = add_random_layer(individual_grammar.copy())
    elif len(individual_grammar) == 3: #qualquer operacao
        random_layer = random.choice(individual_grammar[1:-1].copy())
        new_grammar = random.choice([delete_layer(individual_grammar.copy(), random_layer), replace_n_layers(individual_grammar.copy(), random_layer), clone_layer(individual_grammar.copy(), random_layer)])
    else:
        random_layer = random.choice(individual_grammar[1:-1].copy())
        new_grammar = delete_layer(individual_grammar.copy(), random_layer)

    new_individual.grammar = new_grammar
    new_individual.build_model()
    new_individual.evaluate()
    if new_individual.fitness > individual.fitness:
        return new_individual
    else:
        return individual
