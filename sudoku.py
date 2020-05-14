# Sudoku Solver with GA | Sinan Demir

import random
import numpy as np


class Individual:

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.calc_fitness()

    def calc_fitness(self):
        fitness = 0
        for index in range(9):
            fitness += len(np.unique(self.chromosome[index]))
            fitness += len(np.unique(self.chromosome.transpose()[index]))
        for h in np.hsplit(self.chromosome, 3):
            for v in np.vsplit(h, 3):
                fitness += len(np.unique(v))
        return fitness / 243

    @classmethod
    # To initialize parents.
    def create(cls):
        foo = np.zeros([9, 9], dtype=np.int8)
        for i in range(9):
            genes = list(np.arange(1, 10, dtype=np.int8))
            immutable = [(pos, item) for (pos, item) in enumerate(problem[i]) if problem[i][pos] != 0]
            random.shuffle(genes)
            for pos, item in immutable:
                index = genes.index(item)
                genes[pos], genes[index] = genes[index], genes[pos]
            foo[i] = np.array(genes)
        return foo

    # To create a child.
    def mate(self, other_parent):
        child = np.zeros([9, 9], dtype=np.int8)
        for i in range(9):
            # To cross two parents row.
            if mutation_chance < random.random():
                if .5 < random.random():
                    child[i] = self.chromosome[i]
                else:
                    child[i] = other_parent.chromosome[i]
            # To mutate parent's row.
            else:
                temp = [i for i in self.chromosome[i]]
                immutable = [(pos, item) for (pos, item) in enumerate(temp) if problem[i][pos] != 0]
                random.shuffle(temp)
                for pos, item in immutable:
                    index = temp.index(item)
                    temp[pos], temp[index] = temp[index], temp[pos]
                child[i] = np.array(temp)
        return Individual(child)


# To choose a parent with high fitness.
def roulette_selection():
    sigma = sum([individual.fitness for individual in population[:int(population_size / generation)]])
    pick = random.uniform(0, sigma)
    current = 0
    for individual in population:
        current += individual.fitness
        if current > pick:
            return individual


def show():
    string = ""
    for i in range(9):
        string += "\n"
        if i == 3 or i == 6:
            string += "-" * 7 + "+" + "-" * 8 + "+" + "-" * 7
            string += "\n"
        for j in range(9):
            string += str(population[0].chromosome[i][j]) + " "
            if j == 2 or j == 5:
                string += " | "
    return string


if __name__ == '__main__':
    problem = np.array((0, 0, 6, 0, 0, 0, 0, 0, 0,
                        0, 8, 0, 0, 5, 4, 2, 0, 0,
                        0, 4, 0, 0, 9, 0, 0, 7, 0,
                        0, 0, 7, 9, 0, 0, 3, 0, 0,
                        0, 0, 0, 0, 8, 0, 4, 0, 0,
                        6, 0, 0, 0, 0, 0, 1, 0, 0,
                        2, 0, 3, 0, 0, 0, 0, 0, 1,
                        0, 0, 0, 5, 0, 0, 0, 4, 0,
                        0, 0, 8, 3, 0, 0, 5, 0, 2), dtype=np.int8).reshape(9, 9)
    found = False
    generation = 1
    repetition = 0
    repetition_value = 0
    mutation_chance = 0.2
    population_size = 2000

    print("The population is being created. (Population Size : {})".format(population_size))
    population = list()
    for _ in range(population_size):
        population.append(Individual(Individual.create()))
    while not found and generation != population_size:
        population.sort(key=lambda x: x.fitness, reverse=True)
        if population[0].fitness == 1:
            found = True
            break

        # To prevent the same fitness from coming
        if repetition_value == population[0].fitness:
            repetition += 1
        else:
            repetition_value = population[0].fitness
            repetition = 0

        if repetition == 5:
            repetition = 0
            mutation_chance -= 0.001

        new_generation = list()
        for _ in range(population_size):
            new_generation.append(roulette_selection().mate(roulette_selection()))
        print("\n{}\tGeneration : {:3} | Fitness : {:.5} | Mutation Chance : {:.5}".
              format(show(), generation, population[0].fitness, mutation_chance))
        population = new_generation
        generation += 1

    if not found:
        print("This problem could not be solved.")
    else:
        print("\n{}\tGeneration : {:3} | Fitness : {:.5} | Mutation Chance : {:.5}".
              format(show(), generation, population[0].fitness, mutation_chance))
