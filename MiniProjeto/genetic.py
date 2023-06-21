import numpy as np
def generate_population(pop_size, nlayers, max_nfilters, max_sfilters):
    # Given the parameters returns randomly generated population

    np.random.seed(0)
    pop_nlayers = np.random.randint(1, max_nfilters, (pop_size, nlayers))
    pop_sfilters = np.random.randint(1, max_sfilters, (pop_size, nlayers))
    pop_total = np.concatenate((pop_nlayers, pop_sfilters), axis=1)
    return pop_total

def fitness(pop,X,Y,epochs):
    pop_acc = []
    for i in range(pop.shape[0]):
        nfilters = pop[i][0:3]
        sfilters = pop[i][3:]
        model = CNN(nfilters,sfilters)
        H = model.fit(X,Y,batch_size=32,epochs=epochs)
        acc = H.history['accuracy']
        pop_acc.append(max(acc))
    return pop_acc


def select_parents(pop,nparents,fitness):
    parents = np.zeros((nparents,pop.shape[1]))
    for i in range(nparents):
        best = np.argmax(fitness)
        parents[i] = pop[best]
        fitness[best] = -99999
    return parents

def crossover(parents,pop_size):
    nchild = pop_size - parents.shape[0]
    nparents = parents.shape[0]
    child = np.zeros((nchild,parents.shape[1]))
    for i in range(nchild):
        first = i % nparents
        second = (i+1) % nparents
        child[i,:2] = parents[first][:2]
        child[i,2] = parents[second][2]
        child[i,3:5] = parents[first][3:5]
        child[i,5] = parents[second][5]
    return child


def mutation(child):
    for i in range(child.shape[0]):
        val = np.random.randint(1,6)
        ind = np.random.randint(1,4) - 1
        if child[i][ind] + val > 100:
            child[i][ind] -= val
        else:
            child[i][ind] += val
        val = np.random.randint(1,4)
        ind = np.random.randint(4,7) - 1
        if child[i][ind] + val > 20:
            child[i][ind] -= val
        else:
            child[i][ind] += val
    return child