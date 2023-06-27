import pandas as pd
import torch
#from agent import *
from indiv import *
from grammar import *
import random
import re
import os

#build population of 10 individuals 50% of the population with max hidden layers and 50% with random size
GENERATIONS = 10
POPULATION_SIZE = 10
N_HIDDEN_LAYERS = 2
INPUT_SIZE = 11
OUTPUT_SIZE = 3
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.3


#create statistics dataframe
statistics = pd.DataFrame(columns=['gen', 'best_fitness', 'mean_fitness', 'std_fitness', 'total_cross', 'total_mut', 'mean_size', 'size_best_individual'])

seeds = [29,17,89,56,23,73,12,61,34,98]

if __name__ == '__main__':

    for seed in seeds:
        #build folder seed
        folder = "seed_" + str(seed)
        #create folder
        os.mkdir(folder)
        # STATISTICS
        best_fitness = 0
        mean_fitness = 0
        std_fitness = 0
        total_cross = 0
        total_mut = 0
        mean_size = 0
        size_best_individual = 0
        #define random seed
        random.seed(seed)
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
        best_individual = population[0]
        best_fitness = best_individual.fitness
        mean_fitness = sum([indv.fitness for indv in population])/len(population)
        std_fitness = sum([(indv.fitness - mean_fitness)**2 for indv in population])/len(population)
        size_best_individual = len(best_individual.grammar)
        mean_size = sum([len(indv.grammar) for indv in population])/len(population)


        #Start Generations
        for gen in range(GENERATIONS):
            #add statistics to dataframe
            new_row = pd.DataFrame(
                {'gen': gen, 'best_fitness': best_fitness, 'mean_fitness': mean_fitness, 'std_fitness': std_fitness,
                 'total_cross': total_cross, 'total_mut': total_mut, 'mean_size': mean_size, 'size_best_individual': size_best_individual}, index=[0])
            statistics = pd.concat([statistics, new_row], ignore_index=True)

            #reinicia estatisticas
            best_fitness = 0
            mean_fitness = 0
            std_fitness = 0
            total_cross = 0
            total_mut = 0
            mean_size = 0
            size_best_individual = 0


            new_population = []
            print("Generation: ", gen)
            #print statistics
            print("Best Fitness: ", best_fitness)
            print("Mean Fitness: ", mean_fitness)
            print("Std Fitness: ", std_fitness)
            print("Total Cross: ", total_cross)
            print("Total Mut: ", total_mut)
            print("Best Individual: ", best_individual.grammar)
            #media do tamanho dos individuos
            print("Mean Size: ", sum([len(indv.grammar) for indv in population])/len(population))
            #tamanho do melhor individuo
            print("Size Best Individual: ", len(best_individual.grammar))
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

            #select the best individuals new pop
            new_population.sort(key=lambda x: x.fitness, reverse=True)
            population = new_population[:POPULATION_SIZE]

            if new_population[0].fitness > best_individual.fitness:
                best_individual = new_population[0]
            #add best individual in population
            population[-1] = best_individual

            #STATISTICS
            best_fitness = best_individual.fitness
            mean_fitness = sum([indv.fitness for indv in population])/len(population)
            std_fitness = sum([(indv.fitness - mean_fitness)**2 for indv in population])/len(population)
            size_best_individual = len(best_individual.grammar)
            mean_size = sum([len(indv.grammar) for indv in population]) / len(population)

            best_individual.save_indv(folder, "best_indv_gen_" + str(gen))


        #save statistics
        statistics.to_csv("Statistics/statistics_"+str(seed)+".csv", index=False)
        #evaluate best individual
        teste_indiv = Individual(INPUT_SIZE, 3, OUTPUT_SIZE)
        teste_indiv.load_indv(folder, "best_indv_gen_"+str(GENERATIONS-1))
        teste_indiv.evaluate()
        print("Best Individual Fitness: ", teste_indiv.fitness)
