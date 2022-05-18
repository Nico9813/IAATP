import array
import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import numpy as np

# Cantidad de Ciclos de la Corrida
CANT_CICLOS = 150  # @param {type:"integer"}

# Cantidad de Individuos en la Población
CANT_INDIVIDUOS_POBLACION = 1000  #@param {type:"slider", min:1, max:100, step:1}
CANT_PROPIEDADES = 4

# definimos la función de aptitud a evaluar
def funcAptitud(individuo):
    return [sum(individuo)]

# Inicializa objeto Toolbox auxiliar
toolbox = base.Toolbox()

# indica que los individuos son una lista de genes que aplica la función antes definida
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Attribute generator
toolbox.register("attr_int",   random.randint, -100, 100)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, CANT_PROPIEDADES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pop = toolbox.population(n=CANT_INDIVIDUOS_POBLACION)

for pop_ind in pop:
    print(pop_ind)

# registra la función que se va a evaluar
toolbox.register("evaluate", funcAptitud)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=10)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)

ngen = 100  # Gerations
npop = 1000  # Population

hof = tools.ParetoFront()
stats = tools.Statistics(lambda ind: ind.fitness.values)

# Statistics
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

# Evolution
pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=npop, lambda_=npop,
                                         cxpb=0.7,   mutpb=0.3, ngen=ngen,
                                         stats=stats, halloffame=hof)

best_solution = tools.selBest(pop, 1)[0]
print("")
print("[{}] best_score: {}".format(logbook[-1]['gen'], logbook[-1]['min'][0]))