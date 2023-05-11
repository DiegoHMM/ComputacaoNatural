
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
