import re
import random
from torch import nn
import random


GRAMMAR = [
       # ("Linear",),
        ("Linear", "ReLU"),
        ("Linear", "Tanh"),
        ("Linear", "Sigmoid")
    ]
AVAILABLE_SIZES = [32, 64, 128, 256]


def generate_hidden_layers(input_size, num_hidden_layers, output_size):
    hidden_layers = [(None, input_size)]

    for _ in range(num_hidden_layers):
        layer = random.choice(GRAMMAR)
        hidden_size = random.choice(AVAILABLE_SIZES)
        if len(layer) > 1:
            layer = (layer[0], hidden_size, layer[1])
        else:
            layer = (layer[0], hidden_size)
        hidden_layers.append(layer)
    layer = random.choice(GRAMMAR)
    hidden_layers.append(('Linear', output_size, layer[1]))

    hidden_layers_str = [f"{layer[0]}({hidden_layers[i - 1][1]}, {layer[1]}){' ' + layer[2] if len(layer) > 2 else ''}"
                         for i, layer in enumerate(hidden_layers[1:], start=1)]
    return hidden_layers_str

def build_network(layers):
    module_list = []
    for layer in layers:
        layer_info = layer.split("(")[0]
        sizes = [int(s) for s in layer.split("(")[1].split(")")[0].split(",")]
        if "Linear" in layer_info:
            module_list.append(nn.Linear(*sizes))
        elif " " in layer_info:
            activation_function = getattr(nn, layer_info.split(" ")[1])()
            module_list.append(activation_function)
    return nn.Sequential(*module_list)


def change_layer_dimensions(layer, new_in, new_out):
    # Procura a parte da string que corresponde à dimensões
    layer_parts = re.match(r'Linear\(\d+, \d+\)', layer)
    if layer_parts is not None:
        layer = layer.replace(layer_parts.group(), f'Linear({new_in}, {new_out})')
    return layer

def get_layer_dims(layer):
    # Esta função pega a string de uma camada e extrai as dimensões de entrada e saída
    parts = layer.split('(')[1].split(')')[0].split(',')
    return int(parts[0]), int(parts[1])

def get_activation_function(layer):
    parts = layer.split(' ')
    if len(parts) > 1:
        return parts[1]
    else:
        return ''

def change_activation_function(layer, new_activation):
    parts = layer.split(')')
    if len(parts) > 1:
        layer = layer.replace(parts[1].strip(), new_activation)
    else:
        layer = layer + ' ' + new_activation
    return layer

def get_activation_function(layer):
    parts = layer.split(')')
    if len(parts) > 1:
        return parts[1].strip()
    else:
        return None

def verify_length(individual, max_length=4):
    return max_length - len(individual)

def tournament(population, tournament_size):
    #randomly select two individuals from the population
    tournament = random.sample(population, tournament_size)
    #return the best individual
    return max(tournament, key=lambda x: x.fitness)

def train_individual(individual):
    individual.train()
    return individual
