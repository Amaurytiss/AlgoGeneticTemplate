#########
#Genetic algortihms skeleton
#########

import numpy as np

#######

"""
Here is an example of how you can use this skeleton
We will try to solve this easy problem : find [a,b,c] E [-100,100]^3 such that a^3 - b^2 + c -1 = 0

A chromosome is a list of 3 genes [a,b,c]
The fitness value is the absolute value of a^3 - b^2 + c -1
We seek to minimize the fitness value
We use a discrete representation of the possible alleles with a 0.01 step

"""

######
# PARAMETERS OF THE GENETIC ALGO TO TUNE IN ORDER TO GET BETTER RESULTS

# All the possible alleles a gene can be

POSSIBLE_ALLELES = np.arange(-100,100,0.01)

# Number of genes in a chromosome

N_GENES = 3

# Size of the mating Pool
POOLSIZE = 100

# Number of offsprings of each couple
N_OFFSPRINGS = 100

# If you want to keep the parents in the next generation after reproduction
KEEP_PARENTS = False

# The population size is fixed by the Poolsize and the numbers of offsprings in order to keep the population constant
if KEEP_PARENTS:
    POPULATION_SIZE = POOLSIZE * ( (N_OFFSPRINGS//2) +1 )
else:
    POPULATION_SIZE = POOLSIZE * (N_OFFSPRINGS//2)

# Probability of each gene mutating
PROBA_MUTATION = 0.1

# Number of iterations of the algorithm
GENERATIONS = 10


class Individu:
    
    def __init__(self,genome,n_generation):
        self.generation = n_generation  # Generation of the solution
        self.genome = genome
        self.fitness = self.calculeFitness()

    # Getters and setters
    def getGeneration(self):
        return self.generation

    def getGenome(self):
        return self.genome

    def getFitness(self):
        return self.fitness

    def setFitness(self,fitness_value):
        self.fitness = fitness_value

    # Function to compute the fitness of a solution
    # Here you have to complete with the right function which compute fitness from self.genome
    
    def calculeFitness(self):
        a,b,c = self.getGenome()
        return abs(a**3 - b**2 + c -1)

class Milieu:

    def __init__(self,population):

        self.n_generation = 0                           # The generation number of the population
        self.population = population                    # A list of Individu objects
        self.populationSize = len(population)
        self.bestOffspring = self.population[0]         # Storing the best individu so far, (initialized randomly as the first one)
        self.bestFitness = 100000                           # Storing the fitness of the best individu so far  (initialized as the special value 0 meaning the worst fitness)
        

    #Getters and setters
    def getPopulation(self):
        return self.population
    
    def getPopulationSize(self):
        return self.populationSize

    def getGeneration(self):
        return self.n_generation

    def getBestOffspring(self):
        return self.bestOffspring

    def getBestFitness(self):
        return self.bestFitness

    def incrementGeneration(self):
        self.n_generation += 1

    def setBestOffspring(self,newOffspring):
        self.bestOffspring = newOffspring

    def setBestFitness(self,newFitness):
        self.bestFitness = newFitness

    def setPopulation(self,newPopulation):
        self.population = newPopulation

    # Method to update the best individual so far
    def updateBestFitness(self):
        for indiv in self.getPopulation():
            if indiv.getFitness() < self.getBestFitness():
                self.setBestFitness(indiv.getFitness())
                self.setBestOffspring(indiv)
                print(f' --- Best Fitness has been updated with the value {indiv.getFitness()} --- ')
    
    # Method which return an array of the best individus which are going to reproduce
    def selectMatingPool(self,poolSize):

        if poolSize >= self.getPopulationSize():
            raise ValueError('Poolsize entered is bigger than the population')
        elif poolSize%2!=0:
            raise ValueError('Poolsize should be even')
        else:

            matingPool = []
            fitnessValues = []

            for indiv in self.getPopulation():
                fitnessValues.append(indiv.getFitness())

            fitnessValues = np.array(fitnessValues)
            idxChoisis = list(fitnessValues.argsort()[:poolSize])

            matingPool = np.array(self.getPopulation())[idxChoisis]

        return matingPool
# Method which takes two individus and return a child individu with a chromosome from a crossover of his parents
    def crossover(self,parentA,parentB,n_genes = N_GENES,randomPoint = True):
        
        newChromosome = []
        if randomPoint:
            crossOverPoint = np.random.randint(0,n_genes)

        else:
            crossOverPoint = n_genes//2
        
        choixEchangeGene = np.random.choice([True,False])

        if choixEchangeGene:
            newChromosome = parentA.getGenome()[:crossOverPoint]+parentB.getGenome()[crossOverPoint:]
        else:
            newChromosome = parentB.getGenome()[:crossOverPoint]+parentA.getGenome()[crossOverPoint:]

        return newChromosome
    
    # Method which makes mutations on a chromosome with PROBA_MUTATION on each genes

    def mutation(self,chromosome,proba=PROBA_MUTATION,possible_alleles = POSSIBLE_ALLELES):
        for i in range(len(chromosome)):
            if np.random.random()<PROBA_MUTATION:
                chromosome[i] = np.random.choice(possible_alleles)
        return chromosome

        for i in range(len(chromosome)):
            if np.random.random()<3*PROBA_MUTATION:
                chromosome[i] += np.random.choice([-1,1])*0.01
        return chromosome



    # Method to do the reproduction on the whole population, combining crossover and mutation. Returns a list of child individus
    def reproduction(self,matingPool,n_offsprings):

        if KEEP_PARENTS:
            if len(matingPool)*((n_offsprings/2)+1) > self.populationSize:
                raise ValueError('n_offsprings is too high, population size is going to diverge')

        newPop = []
        np.random.shuffle(matingPool)

        for i_reproduction in range(len(matingPool)//2):
            parentA = matingPool[2*i_reproduction]
            parentB = matingPool[(2*i_reproduction)+1]
            for _ in range(n_offsprings):
                
                chromosomeEnfant = self.crossover(parentA,parentB)  # Crossing over the parents
                chromosomeEnfant = self.mutation(chromosomeEnfant)  # Mutation of the child chromosome
                newPop.append(Individu(chromosomeEnfant,self.getGeneration()))

        if KEEP_PARENTS:    # We add the parents to the new population
            for parent in matingPool:
                newPop.append(parent)

        return newPop

    # Method which simulate the wole generation with selection, reproduction, and all the populations updates
    def simuleGeneration(self):

        matingPool = self.selectMatingPool(poolSize=POOLSIZE)
        newPopulation = self.reproduction(matingPool,n_offsprings=N_OFFSPRINGS)

        self.setPopulation(newPopulation)   # Updating the population
        self.incrementGeneration()          # Increment the generation so we can keep track where we are in the iterations
        self.updateBestFitness()

        print(f"Best solution is from generation {self.getBestOffspring().getGeneration()} with a fitness value of {self.getBestFitness()} and the genome {self.getBestOffspring().getGenome()}")


def main(n_iterations):

# First we initialize the population with random individus
    populationInitiale = []

    for i in range(POPULATION_SIZE):
        genome = [np.random.choice(POSSIBLE_ALLELES) for _ in range(3)] # Here you have to complete with a random genome which fit your problem
        indiv = Individu(genome,0)
        populationInitiale.append(indiv)
    
    # We instanciate the Milieu
    milieu = Milieu(populationInitiale)

    # We loop over the number of generations defined and we simulate the generations
    for i in range(n_iterations):

        print(f"--- GENERATION {i} ---")
        milieu.simuleGeneration()

    # When the loop end we show the best result found
    print(" --- ALGORITHM ENDED --- ")
    print("\n")

    print(f"Best result found is solution : {milieu.getBestOffspring().getGenome()} from generation {milieu.getBestOffspring().getGeneration()} with a fitness value of : {milieu.getBestFitness()}")

if __name__=='__main__':
    main(GENERATIONS)


