import torch
from indiv import *
from grammar import *
import random
import re

 #define random seed
#random.seed(42)

#build population of 10 individuals 50% of the population with max hidden layers and 50% with random size
GENERETAIONS = 10
POPULATION_SIZE = 10
N_HIDDEN_LAYERS = 2
INPUT_SIZE = 11
OUTPUT_SIZE = 3
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.1


#STATISTICS
best_fitness_per_gen = []
mean_fitness_per_gen = []
std_fitness_per_gen = []
total_cross_per_gen = []
total_mut_per_gen = []

best_fitness = 0
mean_fitness = 0
std_fitness = 0
total_cross = 0
total_mut = 0


#teste
individual_1 = Individual(INPUT_SIZE, N_HIDDEN_LAYERS, OUTPUT_SIZE)
child_1 = mutate(individual_1)

print("Individual 1: ", individual_1.grammar)
print("Child 1: ", child_1.grammar)

'''
#Start First Population
population = []
for i in range(POPULATION_SIZE):
    # 50% of the population with max hidden layers
    if random.random() < 0.5:
        population.append(Individual(INPUT_SIZE, N_HIDDEN_LAYERS, OUTPUT_SIZE, "model"+str(i)))
    # 50% of the population with random size
    else:
        population.append(Individual(INPUT_SIZE, random.randint(1, N_HIDDEN_LAYERS), OUTPUT_SIZE, "model"+str(i)))

#train each individual
for indv in population:
    indv.train()

#evaluate each individual
for indv in population:
    indv.evaluate()
#statistics
population.sort(key=lambda x: x.fitness, reverse=True)
best_fitness = population[0].fitness
mean_fitness = sum([indv.fitness for indv in population])/len(population)
std_fitness = sum([(indv.fitness - mean_fitness)**2 for indv in population])/len(population)
best_fitness_per_gen.append(best_fitness)
mean_fitness_per_gen.append(mean_fitness)
std_fitness_per_gen.append(std_fitness)
total_cross_per_gen.append(total_cross)
total_mut_per_gen.append(total_mut)


#Start Generations
for gen in range(GENERETAIONS):
    new_population = []
    print("Generation: ", gen)
    #print statistics
    print("Best Fitness: ", best_fitness)
    print("Mean Fitness: ", mean_fitness)
    print("Std Fitness: ", std_fitness)
    print("Total Cross: ", total_cross)
    print("Total Mut: ", total_mut)
    print("")

    # Mutate and crossover
    while len(new_population) < len(population):
        #seleciona 2 individuos
        indv1 = random.choice(population)
        indv2 = random.choice(population)
        #crossover
        if random.random() < CROSSOVER_RATE:
            #crossover
            new_indv1, new_indv2 = point_crossover(indv1, indv2)
            total_cross += 1
            #mutacao
            if random.random() < MUTATION_RATE:
                new_indv1 = mutate(new_indv1)
                total_mut += 1
            if random.random() < MUTATION_RATE:
                new_indv2 = mutate(new_indv2)
                total_mut += 1
            #adiciona os novos individuos a nova populacao
            new_population.append(new_indv1)
            new_population.append(new_indv2)
        #mutate
        if random.random() < MUTATION_RATE:
            new_indv1 = mutate(indv1)
            new_population.append(new_indv1)
            total_mut += 1

        else:
            new_population.append(indv1)
            new_population.append(indv2)

    #evaluate new population
    for indv in new_population:
        indv.evaluate()

    #select the best individuals
    new_population.sort(key=lambda x: x.fitness, reverse=True)
    population = new_population[:POPULATION_SIZE]
    #STATISTICS
    best_fitness = population[0].fitness
    mean_fitness = sum([indv.fitness for indv in population])/len(population)
    std_fitness = sum([(indv.fitness - mean_fitness)**2 for indv in population])/len(population)
    total_cross_per_gen.append(total_cross)
    total_mut_per_gen.append(total_mut)
    best_fitness_per_gen.append(best_fitness)
    mean_fitness_per_gen.append(mean_fitness)
    std_fitness_per_gen.append(std_fitness)
'''