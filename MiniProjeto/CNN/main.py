import pandas as pd
import torch
#from agent import *
from indiv import *
from grammar import *
import random
import re

 #define random seed
random.seed(42)

#build population of 10 individuals 50% of the population with max hidden layers and 50% with random size
GENERATIONS = 10
POPULATION_SIZE = 5
N_HIDDEN_LAYERS = 1
INPUT_SIZE = 11
OUTPUT_SIZE = 3
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.3


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

#create statistics dataframe
statistics = pd.DataFrame(columns=['gen', 'best_fitness', 'mean_fitness', 'std_fitness', 'total_cross', 'total_mut'])

if __name__ == '__main__':

    #Start First Population
    population = []
    for i in range(POPULATION_SIZE):
        # 50% of the population with max hidden layers
        if random.random() < 0.5:
            population.append(Individual(INPUT_SIZE, N_HIDDEN_LAYERS, OUTPUT_SIZE))
        # 50% of the population with random size
        else:
            population.append(Individual(INPUT_SIZE, random.randint(0, N_HIDDEN_LAYERS), OUTPUT_SIZE))

    #train each individual
    for indv in population:
        indv.train()

    #evaluate each individual
    for i, indv in enumerate(population):
        print("Individual: ", indv.grammar)
        print(str(i) + "/" + str(len(population)))
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
    for gen in range(GENERATIONS):
        #add statistics to dataframe
        new_row = pd.DataFrame(
            {'gen': gen, 'best_fitness': best_fitness, 'mean_fitness': mean_fitness, 'std_fitness': std_fitness,
             'total_cross': total_cross, 'total_mut': total_mut}, index=[0])
        statistics = pd.concat([statistics, new_row], ignore_index=True)

        new_population = []
        print("Generation: ", gen)
        #print statistics
        print("Best Fitness: ", best_fitness)
        print("Mean Fitness: ", mean_fitness)
        print("Std Fitness: ", std_fitness)
        print("Total Cross: ", total_cross)
        print("Total Mut: ", total_mut)
        print("Best Individual: ", population[0].grammar)
        #media do tamanho dos individuos
        print("Mean Size: ", sum([len(indv.grammar) for indv in population])/len(population))
        #Quantidade de individuos iguais (individuos com o mesmo grammar)
        #print("Same Individuals: ", len(population) - len(set([indv.grammar for indv in population])))

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
        for i, indv in enumerate(new_population):
            print("Individual: ", indv.grammar)
            print(str(i)+"/"+str(len(new_population)))
            indv.train()
        for indv in new_population:
            indv.evaluate()

        #Elistism
        best_individual = population[0]
        new_population.append(best_individual)
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
        best_individuo = population[0]
        best_individuo.save_indv("best_indv_gen_" + str(gen))
        

    #save statistics
    statistics.to_csv("Statistics/statistics.csv", index=False)
    #evaluate best individual
    teste_indiv = Individual(INPUT_SIZE, 3, OUTPUT_SIZE)
    teste_indiv.load_indv("best_indv_gen_"+str(GENERATIONS-1))
    teste_indiv.evaluate()
    print("Best Individual Fitness: ", teste_indiv.fitness)
