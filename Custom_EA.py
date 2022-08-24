from matplotlib import pyplot as plt
import time, math, random, bisect
from scipy.sparse import csr_matrix
import gym
import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


class NN_agent :
    
    def __init__(self, n_nodes):     

        self.fitness = 0
        self.nodes = n_nodes
        self.weights = []
        self.biases = []
        for i in range(len(n_nodes) - 1):
            self.weights.append(np.random.uniform(low=-1, high=1, size=(n_nodes[i], n_nodes[i+1])).tolist())
            self.biases.append(np.random.uniform(low=-1, high=1, size=(n_nodes[i+1])).tolist())

  
    def getOutput(self, input):

        output = input
        for i in range(len(self.nodes)-1):
            output = np.reshape(np.matmul(output, self.weights[i]) + self.biases[i], (self.nodes[i+1]))
        return np.argmax(sigmoid(output))

class Population :

    def __init__(self, populationCount, p_mut, n_nodes):
        self.nodeCount = n_nodes
        self.popCount = populationCount
        self.m_rate = p_mut
        self.population = [NN_agent(n_nodes) for i in range(populationCount)]


    def createOffspring(self, nn1, nn2):
        
        offspring = NN_agent(self.nodeCount)

        for i in range(len(offspring.weights)):
            for j in range(len(offspring.weights[i])):
                for k in range(len(offspring.weights[i][j])):
                    if random.random() < self.m_rate:
                        offspring.weights[i][j][k] = random.uniform(-1, 1)
                    else:
                        offspring.weights[i][j][k] = (nn1.weights[i][j][k] + nn2.weights[i][j][k])/2.0

        for i in range(len(offspring.biases)):
            for j in range(len(offspring.biases[i])):
                if random.random() < self.m_rate:
                    offspring.biases[i][j] = random.uniform(-1, 1)
                else:
                    offspring.biases[i][j] = (nn1.biases[i][j] + nn2.biases[i][j])/2.0

        return offspring


    def createNewGeneration(self):       
        nextGen = []
        fitnessSum = [0]
        for i in range(len(self.population)):
            fitnessSum.append(fitnessSum[i]+self.population[i].fitness)
        
        while(len(nextGen) < self.popCount):
            r1 = random.uniform(0, fitnessSum[len(fitnessSum)-1] )
            r2 = random.uniform(0, fitnessSum[len(fitnessSum)-1] )
            nn1 = self.population[bisect.bisect_right(fitnessSum, r1)-1]
            nn2 = self.population[bisect.bisect_right(fitnessSum, r2)-1]
            nextGen.append( self.createOffspring(nn1, nn2) )
        self.population.clear()
        self.population = nextGen

MAX_GENERATIONS = 30
MAX_STEPS = 500 
POPULATION_COUNT = 30
MUTATION_RATE = 0.05


env = gym.make('CartPole-v1')
observation = env.reset()

dim_in = env.observation_space.shape[0]
dim_out = env.action_space.n
pop = Population(POPULATION_COUNT, MUTATION_RATE, [dim_in, 8, 8, dim_out])

bestNeuralNets = []
MAXFIT = []
AVGFIT = []

for gen in range(MAX_GENERATIONS):
    genAvgFit = 0.0
    maxFit = 0.0
    maxNeuralNet = None
    for nn in pop.population:
        totalReward = 0
        
        for step in range(MAX_STEPS):
            env.render()
            action = nn.getOutput(observation)
            observation, reward, done, info = env.step(action)
            totalReward += reward
            if done:
                observation = env.reset()
                break
        nn.fitness = totalReward
        genAvgFit += nn.fitness
        if nn.fitness > maxFit :
            maxFit = nn.fitness
            maxNeuralNet = nn

    bestNeuralNets.append(maxNeuralNet)
    genAvgFit/=pop.popCount
    print("Generation : %3d |  Avg Fitness : %4.0f  |  Max Fitness : %4.0f  " % (gen+1, genAvgFit, maxFit) )
    MAXFIT.append(maxFit) 
    AVGFIT.append(genAvgFit)
    pop.createNewGeneration()
        
env.close()

plt.figure();
plt.plot(range(MAX_GENERATIONS), AVGFIT)
plt.plot(range(MAX_GENERATIONS), MAXFIT)
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend(['Mean fitness', 'Max fitness'])
plt.show()