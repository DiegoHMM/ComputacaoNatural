import torch
from indiv import Individual
from grammar import *
import re
#Gramatica


#build population of 10 individuals
population = []
N_HIDDEN_LAYERS = 4
INPUT_SIZE = 11
OUTPUT_SIZE = 3

ind = Individual(INPUT_SIZE, N_HIDDEN_LAYERS, OUTPUT_SIZE, "model")
ind2 = Individual(INPUT_SIZE, N_HIDDEN_LAYERS, OUTPUT_SIZE, "model")

print("Individual 1: ", ind.grammar)
#print("Individual 2: ", ind2.grammar)


mut_ind = mutate(ind.grammar)

print("Individual mutado: ",mut_ind)

# Encontra uma sequência compatível na segunda lista
#child_1, child_2 = point_crossover(ind.grammar, ind2.grammar)

#print(child_1)
#print(child_2)








#['Linear(11, 128) Tanh', 'Linear(128, 256) Sigmoid', 'Linear(256, 256)', 'Linear(256, 3)']
#['Linear(11, 128) Sigmoid', 'Linear(128, 128) Tanh', 'Linear(128, 64) ReLU', 'Linear(64, 256) Tanh', 'Linear(256, 3)']


"""
for i in range(1):
    population.append(Individual(INPUT_SIZE, OUTPUT_SIZE, N_HIDDEN_LAYERS, "model"+str(i)))

#train each individual
for indv in population:
    indv.train()

#evaluate each individual
for indv in population:
    indv.evaluate()

#sort population by fitness
population.sort(key=lambda x: x.fitness, reverse=True)
#print best individual
print(population[0].fitness)
#save best individual
population[0].save_indv("best_model")

#load model and evaluate it
indv = Individual(11, 3, (1,3), (64, 512), "best_model")
indv.load_indv("best_model")
indv.evaluate()
"""








