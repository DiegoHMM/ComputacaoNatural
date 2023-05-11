import random
from Node import Node, TERMINALS, OPERATORS
import pandas as pd
import numpy as np
from Operadores import *
from Node import *


def calculate_fitness(individual,X, y):
        normalize = y - np.mean(y)
        normalize = np.power(normalize, 2)
        normalize = np.sum(normalize)
        real_diff = list()
        y_pred = np.array([individual.evaluate(*x) for x in X])
        for i in range(len(y)):
            real_diff.append(np.power(y[i] - y_pred[i], 2))
        if float('inf') in real_diff:
            return float('inf')
        else:
            # return np.sqrt(np.sum(real_diff))
            return np.sqrt(np.sum(real_diff) / normalize)


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
def trees_are_different(tree1, tree2):
    if tree1.type != tree2.type or tree1.value != tree2.value:
        return True

    if tree1.type == "terminal":
        return tree1.const_value != tree2.const_value if tree1.value == "const" else False

    if tree1.type == "operator":
        if trees_are_different(tree1.left, tree2.left):
            return True
        if tree1.right and tree2.right and trees_are_different(tree1.right, tree2.right):
            return True

    return False

def mutate(tree):
    max_depth = 7
    total_nodes = tree.count_nodes()
    while(True):
        random_index = random.randint(1, total_nodes-1)
        new_subtree = create_random_tree(max_depth - tree.node_depth())
        if new_subtree is not None:
            new_tree = tree.copy()
            new_tree.replace_subtree(random_index, new_subtree)
            if new_tree.node_depth() <= max_depth:
                return new_tree

def get_nodes_at_depth(node, depth, current_depth=0):
    if current_depth == depth:
        return [node]
    nodes = []
    if node.left is not None:
        nodes.extend(get_nodes_at_depth(node.left, depth, current_depth + 1))
    if node.right is not None:
        nodes.extend(get_nodes_at_depth(node.right, depth, current_depth + 1))
    return nodes



def crossover(parent1, parent2, max_depth=7):
    attempts = 0
    max_attempts = 100
    while attempts < max_attempts:
        attempts += 1

        parent1_node_count = parent1.count_nodes()
        parent2_node_count = parent2.count_nodes()

        parent1_subtree_index = random.randint(1, parent1_node_count - 1)
        parent2_subtree_index = random.randint(1, parent2_node_count - 1)

        parent1_subtree = parent1.select_subtree(parent1_subtree_index)
        parent2_subtree = parent2.select_subtree(parent2_subtree_index)

        if parent1_subtree == parent2_subtree:
            continue

        offspring1 = parent1.copy()
        offspring2 = parent2.copy()

        offspring1.replace_subtree(parent1_subtree_index, parent2_subtree.copy())
        offspring2.replace_subtree(parent2_subtree_index, parent1_subtree.copy())

        offspring1_depth = offspring1.get_depth()
        offspring2_depth = offspring2.get_depth()

        if offspring1_depth > max_depth or offspring2_depth > max_depth:
            continue
        
        return offspring1, offspring2

    return parent1, parent2

def crossover_elite(parent1, parent2, X, y, max_depth=7):
    parent1_fitness = calculate_fitness(parent1, X, y)
    parent2_fitness = calculate_fitness(parent2, X, y)

    attempts = 0
    max_attempts = 100

    while attempts < max_attempts:
        offspring1, offspring2 = crossover(parent1, parent2, max_depth=7)

        offspring1_fitness = calculate_fitness(offspring1, X, y)
        offspring2_fitness = calculate_fitness(offspring2, X, y)

        # Check if offspring fitness is not worse than their parents' fitness
        if offspring1_fitness < parent1_fitness and offspring2_fitness < parent2_fitness:
            return offspring1, offspring2

        attempts += 1

    # If no better offspring is found after max_attempts, return the best offspring found
    return offspring1, offspring2

def epsilon_lexicase_selection(population, train_cases, train_labels, mad):
    candidates = list(range(len(population)))
    np.random.shuffle(candidates)
    cases = list(zip(train_cases, train_labels))
    np.random.shuffle(cases)

    while len(candidates) > 1 and cases:
        x, y = cases.pop(0)
        errors = [abs(population[i].evaluate(*x) - y) for i in candidates]
        min_error = min(errors)
        median_error = np.median(errors)
        epsilon_threshold = min_error + mad

        survivors = [candidates[i] for i in range(len(candidates)) if errors[i] <= epsilon_threshold]
        candidates = survivors

    return population[candidates[-1]]  # Retorna o último indivíduo restante


def roulette_selection(population, fitnesses):
    # Calcular a soma total dos fitnesses inversos
    total_fitness = sum(1 / fitness for fitness in fitnesses)

    # Gerar um número aleatório entre 0 e a soma total dos fitnesses
    random_threshold = random.uniform(0, total_fitness)

    # Percorrer a população acumulando a soma dos fitnesses inversos
    cumulative_sum = 0
    for i in range(len(population)):
        cumulative_sum += 1 / fitnesses[i]
        if cumulative_sum >= random_threshold:
            return population[i]

    # Retornar o último indivíduo se a condição nunca for satisfeita
    return population[-1]


def tournament_selection(population, fitnesses, tournament_size):
    selected_indices = random.sample(range(len(population)), tournament_size)
    selected_fitnesses = [fitnesses[i] for i in selected_indices]
    best_index = selected_indices[np.argmin(selected_fitnesses)]
    return population[best_index]


def mad_of_errors(predictions, labels):
    errors = np.abs(predictions - labels[np.newaxis, :])
    median_error = np.median(errors)
    mad = np.median(np.abs(errors - median_error))
    return mad

def evolve(selection_type, population, train_cases, train_labels, fitnesses, generations, mutation_rate=0.2, crossover_rate=0.6, elitism=False, tournament_size=3):

    stats = {
        'generation': [],
        'population_size': [],
        'best_fitness': [],
        'avg_fitness': [],
        'worst_fitness': [],
        'duplicates': [],
        'mutations': [],
        'crossovers': []
    }
    for gen in range(generations):
        new_population = []
        n_mut = 0
        n_cross = 0

        if elitism:
            best_individual_index = np.argmin(fitnesses)
            best_individual = population[best_individual_index]
            new_population.append(best_individual)
        
        duplicates = count_duplicates(population)

        while len(new_population) < (len(population) - 1 if elitism else len(population)):
            if selection_type == 'tournament':
                parent1 = tournament_selection(population, fitnesses, tournament_size)
                parent2 = tournament_selection(population, fitnesses, tournament_size)
            elif selection_type == 'roulette':
                parent1 = roulette_selection(population, fitnesses)
                parent2 = roulette_selection(population, fitnesses)
            else: #Lexicase
                pop_size = len(population)
                n_cases = len(train_cases)
                predictions = np.array([[individual.evaluate(*x) for x in train_cases] for individual in population])
                mad = mad_of_errors(predictions, train_labels)

                parent1 = epsilon_lexicase_selection(population, train_cases, train_labels, mad)
                parent2 = epsilon_lexicase_selection(population, train_cases, train_labels, mad)

            # Crossover sem elite
            offspring1, offspring2 = parent1, parent2
            if random.random() < crossover_rate:
                offspring1, offspring2 = crossover(parent1, parent2)
                n_cross += 1

            #Crossover elite
            #offspring1, offspring2 = parent1, parent2
            #if random.random() < crossover_rate:
            #    offspring1, offspring2 = crossover_elite(parent1, parent2, train_cases, train_labels)
            #    #if offspring1 not in new_population and offspring2 not in new_population:
            #    n_cross += 1

            # Mutation
            if random.random() < mutation_rate:
                mutated1 = mutate(offspring1)
                mutated2 = mutate(offspring2)
                offspring1 = mutated1
                offspring2 = mutated2
                n_mut += 1


            new_population.extend([offspring1, offspring2])
      
        # Update fitness of individuals
        fitnesses = [calculate_fitness(individual, train_cases, train_labels) for individual in new_population]
        
        # Replace the old population with the new one
        population = new_population

        #Estatisticas
        stats['generation'].append(gen)
        stats['population_size'].append(len(population))
        stats['best_fitness'].append(np.min(fitnesses))
        stats['avg_fitness'].append(np.mean(fitnesses))
        stats['worst_fitness'].append(np.max(fitnesses))
        stats['duplicates'].append(duplicates)
        stats['mutations'].append(n_mut)
        stats['crossovers'].append(n_cross)
        #print gen
        print("Tamanho da populacao: ", len(population))
        print("Generation: ", gen)
        print("Best fitness: ", np.min(fitnesses))
        print("Media fitness: ", np.mean(fitnesses))
        #Printa melhor indv e a fitness
        best_individual = population[np.argmin(fitnesses)]
        print_tree(best_individual)

    return pd.DataFrame(stats), best_individual



def get_data(path):
    df_train = pd.read_csv(path, header=None)
    # X = todas as colunas exceto a última; y = última coluna
    X = df_train.iloc[:, :-1].values
    y = df_train.iloc[:, -1].values
    return X, y

#calcula quantidade de indivíduos iguais na população
def count_duplicates(population):
    unique_individuals = set(population)
    num_duplicates = len(population) - len(unique_individuals)
    return num_duplicates

def create_initial_population(pop_size, max_depth, terminal_prob=0.5):
    population = []
    while len(population) < pop_size:
        individual = Node.ramped_half(max_depth, terminal_prob, min_size=3)
        population.append(individual)
    return population

def create_random_tree(max_depth=7, terminal_prob=0.3):
    if max_depth == 0 or random.random() < terminal_prob:
        return Node("terminal", random.choice(TERMINALS))
    else:
        value = random.choice(OPERATORS)
        left = create_random_tree(max_depth - 1, terminal_prob)
        right = create_random_tree(max_depth - 1, terminal_prob)
        return Node("operator", value, left, right)
