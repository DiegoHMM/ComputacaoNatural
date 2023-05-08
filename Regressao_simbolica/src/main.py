import random
import numpy as np
import pandas as pd

from Node import Node
from Operadores import *
from config import run_exp
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    seeds = [100, 256, 345, 789, 901, 1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9876, 8765, 7654, 6543, 5432, 4321, 3210, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1]
    #Parâmetros gerais
    max_depth = 7
    train_path = 'datasets\synth1\synth1-train.csv'
    test_path = 'datasets\synth1\synth1-test.csv'
    n_runs = 30


    # Define your parameters
    parametros_1 = {
    "elitism": True,
    "pop_size": 100,
    "generations": 50,
    "mutation_rate": 0.15,
    "crossover_rate": 0.6,
    "selection_type": 'tournament',
    "tournament_size": 2,
    }

    parametros_2 = parametros_1.copy()
    parametros_2['selection_type'] = 'roulette'

    parametros_3 = parametros_1.copy()
    parametros_3['selection_type'] = 'epsilon_lexicase'


    all_stats_2, best_run_2 = run_exp(parametros_2, train_path, test_path, n_runs, seeds)
    general_stats_2 = pd.concat(all_stats_2, ignore_index=True)

    pd.DataFrame(general_stats_2).to_csv('Estatisticas\stats_exp_roulette.csv')

#run main
if __name__ == "__main__":
    main()