import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import numpy as np

from tetris import getScore

import ag_constants

# Create a minimizing fitness function
# Cantidad de Ciclos de la Corrida
CANT_CICLOS = 150  # @param {type:"integer"}

# Cantidad de Individuos en la Población
# @param {type:"slider", min:1, max:100, step:1}
CANT_INDIVIDUOS_POBLACION = 100
CANT_PROPIEDADES = 4

# Inicializa objeto Toolbox auxiliar
toolbox = base.Toolbox()

# indica que los individuos son una lista de genes que aplica la función antes definida
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Attribute generator
toolbox.register("attr_int", random.randint, -1000, 1000)

# Structure initializers
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_int, CANT_PROPIEDADES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pop = toolbox.population(n=ag_constants.npop)

print(len(pop))

# registra la función que se va a evaluar
toolbox.register("evaluate", getScore)
toolbox.register("select", tools.selTournament, tournsize=10)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)

hof = tools.ParetoFront()
stats = tools.Statistics(lambda ind: ind.fitness.values)

# Statistics
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

# Evolution
pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=ag_constants.npop, lambda_=ag_constants.npop,
                                         cxpb=0.7,   mutpb=0.3, ngen=ag_constants.ngen,
                                         stats=stats, halloffame=hof)

best_solution = tools.selBest(pop, 1)[0]

print("Best solution: %s" % best_solution)
print("[{}] best_score: {}".format(
    logbook[-1]['gen'], logbook[-1]['min'][0]))
input("Press Enter to continue...")
