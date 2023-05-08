
import random
import numpy as np
import pandas as pd
#Local
from Operadores import *
from Node import Node

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


#calcula quantidade de indivíduos iguais na população
def count_duplicates(population):
    unique_individuals = set(population)
    num_duplicates = len(population) - len(unique_individuals)
    return num_duplicates

def create_initial_population(pop_size, max_depth, terminal_prob):
    population = []
    while len(population) < pop_size:
        individual = Node.ramped_half(max_depth, terminal_prob, min_size=3)
        population.append(individual)
    return population

def mad_of_errors(predictions, labels):
    errors = np.abs(predictions - labels[np.newaxis, :])
    median_error = np.median(errors)
    mad = np.median(np.abs(errors - median_error))
    return mad

def evolve(selection_type, population, train_cases, train_labels, fitnesses, generations, max_depth, mutation_rate=0.2, crossover_rate=0.6, elitism=False, tournament_size=3):

    l_indv_iguais = []
    l_mutacoes = []
    l_crossovers = []
    l_avg_fitness = []
    l_best_fitness = []
    l_pior_fitness = []

    for gen in range(generations):
        new_population = []
        n_mut = 0
        n_cross = 0

        l_indv_iguais.append(count_duplicates(population))
        print("Total individuos iguais: ", count_duplicates(population))

        if elitism:
            # Select the best individual from the current population
            best_individual_index = np.argmin(fitnesses)
            best_individual = population[best_individual_index]
            new_population.append(best_individual)

        while len(new_population) < (len(population) - 1 if elitism else len(population)):
            if selection_type == 'tournament':
                # Parent selection
                parent1 = tournament_selection(population, fitnesses, tournament_size)
                parent2 = tournament_selection(population, fitnesses, tournament_size)
            elif selection_type == 'roulette':
                parent1 = roulette_selection(population, fitnesses)
                parent2 = roulette_selection(population, fitnesses)
            elif selection_type == 'epsilon_lexicase':
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
        l_indv_iguais.append(count_duplicates(population))
        l_mutacoes.append(n_mut)
        l_crossovers.append(n_cross)
        l_avg_fitness.append(np.mean(fitnesses))
        l_best_fitness.append(np.min(fitnesses))
        l_pior_fitness.append(np.max(fitnesses))
        #Printa melhor indv e a fitness
        best_individual = population[np.argmin(fitnesses)]
        #print gen
        print("Tamanho da populacao: ", len(population))
        print("Generation: ", gen)
        print("Best fitness: ", np.min(fitnesses))
        print("Media fitness: ", np.mean(fitnesses))
        print_tree(best_individual)

    return population, fitnesses, best_individual, l_indv_iguais, l_mutacoes, l_crossovers, l_avg_fitness, l_best_fitness, l_pior_fitness

