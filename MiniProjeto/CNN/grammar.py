import re
import random
from torch import nn
import random


GRAMMAR = [
        ("Linear",),
        ("Linear", "ReLU"),
        ("Linear", "Tanh"),
        ("Linear", "Sigmoid")
    ]
AVAILABLE_SIZES = [64, 128, 256, 512]


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

    hidden_layers.append(('Linear', output_size))

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
            if " " in layer_info:
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

def point_crossover(individual_1, individual_2):
    random_layer = random.choice(individual_1)
    random_index_layer = individual_1.index(random_layer)
    print("Random layer: ", random_layer)
    # Esta função encontra uma sequência de camadas que são compatíveis com a camada inicial
    start_in_dim, start_out_dim = get_layer_dims(random_layer)
    # Encontra as camadas que têm a mesma dimensão de entrada que a camada inicial
    possible_starts = [layer for layer in individual_2 if get_layer_dims(layer)[0] == start_in_dim]

    for random_layer in possible_starts:
        sequence = [random_layer]
        next_in_dim = get_layer_dims(random_layer)[1]
        # Continua adicionando camadas à sequência enquanto a próxima dimensão de entrada puder ser encontrada
        while next_in_dim != start_out_dim:
            next_layers = [layer for layer in individual_2 if get_layer_dims(layer)[0] == next_in_dim]
            if not next_layers:
                break
            next_layer = random.choice(next_layers)
            sequence.append(next_layer)
            next_in_dim = get_layer_dims(next_layer)[1]
        # Se a sequência terminar com a dimensão de saída correta, retorna a sequência
        if next_in_dim == start_out_dim:
            print("Sequence: ", sequence)
            individual_1[random_index_layer:random_index_layer + 1] = sequence
            #substitui sequencia do individuo 2 pela random_layer
            individual_2[individual_2.index(sequence[0]):individual_2.index(sequence[-1]) + 1] = [random_layer]
            return individual_1, individual_2
    # Se nenhuma sequência compatível for encontrada, retorna None
    return None, None


def verify_length(individual, max_length=6):
    return max_length - len(individual)

def delete_layer(individual, random_layer):
    prev_layer = individual[individual.index(random_layer) - 1]
    next_layer = individual[individual.index(random_layer) + 1]
    #get dimensions of random_layer
    in_dim, out_dim = get_layer_dims(random_layer)
    #change dimensions of prev layer
    new_prev_layer = change_layer_dimensions(prev_layer, get_layer_dims(prev_layer)[0], in_dim)
    #change dimensions of next layer
    new_next_layer = change_layer_dimensions(next_layer, in_dim, get_layer_dims(next_layer)[1])
    #change layer of individual
    individual[individual.index(random_layer)-1] = new_prev_layer
    individual[individual.index(random_layer)+1] = new_next_layer
    individual.remove(random_layer)
    return individual


def replace_n_layers(individual, random_layer):
    prev_index_layer = individual.index(random_layer) - 1
    next_index_layer = individual.index(random_layer) + 1
    next_in_dim = get_layer_dims(individual[next_index_layer])[0]
    layer = random.choice(GRAMMAR)
    # Gera a primeira camada com saída igual a next_in_dim
    first_hidden_size = next_in_dim
    layer = layer[:1] + (first_hidden_size,) + layer[1:]
    # Gera as camadas intermediárias
    n_new_layers = random.randint(1, max(1, verify_length(individual)) - 1)
    for _ in range(n_new_layers):
        hidden_size = random.choice(AVAILABLE_SIZES)
        layer = layer[:1] + (hidden_size,) + layer[1:]
    # Gera a última camada com entrada igual a next_in_dim
    last_hidden_size = next_in_dim
    layer = layer[:1] + layer[1:] + (last_hidden_size,)
    individual.insert(next_index_layer, layer)
    return individual

def clone_layer(individual, random_layer):
    random_index_layer = individual.index(random_layer)
    _, out_dim = get_layer_dims(random_layer)
    new_layer = change_layer_dimensions(random_layer, out_dim, out_dim)
    print(new_layer)
    individual.insert(random_index_layer + 1, new_layer)
    return individual

def mutate(individual):
    random_layer = random.choice(individual[1:-1])
    new_individual = delete_layer(individual, random_layer)
    #if verify_length(individual) > 0: #Nao pode deletar
    #    new_individual = random.choice[replace_layer(individual, random_layer), clone_layer(individual, random_layer)]
    #if verify_length(individual) > 1: #qualquer operacao
    #    new_individual = random.choice[delete_layer(individual, random_layer), replace_layer(individual, random_layer), clone_layer(individual, random_layer)]
    #else:
    #    new_individual = delete_layer(individual, random_layer)

    return new_individual