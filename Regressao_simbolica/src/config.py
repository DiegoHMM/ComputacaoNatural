from Operadores import get_data, create_initial_population, calculate_fitness, evolve
import random

def run_exp(parameters, train_path, test_path, n_runs, seeds):
    #Unpack parameters
    elitism = parameters['elitism']
    pop_size = parameters['pop_size']
    generations = parameters['generations']
    mutation_rate = parameters['mutation_rate']
    crossover_rate = parameters['crossover_rate']
    selection_type = parameters['selection_type']
    tournament_size = parameters['tournament_size']
    #Initialize variables
    all_stats = []
    best_test = 999999
    #Read data
    X_train, y_train = get_data(train_path)
    X_test, y_test = get_data(test_path)
    for i in range(n_runs):
        print("Run: ", i)
        #define random seed
        random.seed(seeds[i])
        #Create population
        initial_population = create_initial_population(pop_size, max_depth=7)
        fitnesses = [calculate_fitness(individual, X_train, y_train) for individual in initial_population]
        #Evolve pop
        stats, best_indv = evolve(selection_type, initial_population,X_train, y_train, fitnesses, generations, mutation_rate, crossover_rate, elitism, tournament_size)
        #Test
        test_fitness = calculate_fitness(best_indv, X_test, y_test)
        stats['test_fitness'] = test_fitness
        if test_fitness < best_test:
            best_test = test_fitness
            best_run = stats
        all_stats.append(stats)
    return all_stats, best_run