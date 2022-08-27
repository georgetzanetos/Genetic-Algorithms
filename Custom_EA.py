import gym
from matplotlib import pyplot as plt
import random, bisect
import numpy as np

#sigmoid activation function
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


class NN_agent :
    
    def __init__(self, n_nodes):     

        self.fitness = 0
        self.nodes = n_nodes
        self.weights = []
        self.biases = []
        for i in range(len(n_nodes) - 1):
            # np.random.seed(1)
            self.weights.append(np.random.uniform(low=-1, high=1, size=(n_nodes[i], n_nodes[i+1])).tolist())
            # np.random.seed(2)
            self.biases.append(np.random.uniform(low=-1, high=1, size=(n_nodes[i+1])).tolist())

    #evaluation
    def getOutput(self, input):

        output = input
        for i in range(len(self.nodes)-1):
            output = np.reshape(np.matmul(output, self.weights[i]) + self.biases[i], (self.nodes[i+1]))
        return np.argmax(sigmoid(output))

class Population :

    def __init__(self, n_individuals, p_mut, n_nodes):
        self.nodeCount = n_nodes
        self.popCount = n_individuals
        self.m_rate = p_mut
        self.population = [NN_agent(n_nodes) for i in range(n_individuals)]

    #variation
    def createOffspring(self, nn1, nn2):
        
        offspring = NN_agent(self.nodeCount)

        for i in range(len(offspring.weights)):
            for j in range(len(offspring.weights[i])):
                for k in range(len(offspring.weights[i][j])):
                    # random.seed(2)
                    if random.random() < self.m_rate:
                        # random.seed(3)
                        offspring.weights[i][j][k] = random.uniform(-1, 1)
                    else:
                        offspring.weights[i][j][k] = (nn1.weights[i][j][k] + nn2.weights[i][j][k])/2.0

        for i in range(len(offspring.biases)):
            for j in range(len(offspring.biases[i])):
                # random.seed(4)
                if random.random() < self.m_rate:
                    # random.seed(5)
                    offspring.biases[i][j] = random.uniform(-1, 1)
                else:
                    offspring.biases[i][j] = (nn1.biases[i][j] + nn2.biases[i][j])/2.0

        return offspring

    #selection
    def newGen(self):       
        total_fitness = [0]
        next_gen = []
        for i in range(len(self.population)):
            total_fitness.append(total_fitness[i]+self.population[i].fitness)
        
        while(len(next_gen) < self.popCount):
            # random.seed(6)
            r1 = random.uniform(0, total_fitness[len(total_fitness)-1] )
            # random.seed(7)
            r2 = random.uniform(0, total_fitness[len(total_fitness)-1] )
            nn1 = self.population[bisect.bisect_right(total_fitness, r1)-1]
            nn2 = self.population[bisect.bisect_right(total_fitness, r2)-1]
            next_gen.append(self.createOffspring(nn1, nn2))
        self.population.clear()
        self.population = next_gen

MAX_GENERATIONS = 30
MAX_STEPS = 500 
POPULATION_COUNT = 30
MUTATION_RATE = 0.02

env = gym.make('CartPole-v1')
observation = env.reset()

dim_in = env.observation_space.shape[0]
dim_out = env.action_space.n
pop = Population(POPULATION_COUNT, MUTATION_RATE, [dim_in, 8, 8, dim_out])

MAXFIT = []
AVGFIT = []

for gen in range(MAX_GENERATIONS):
    max = 0
    avg = 0
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
        avg += nn.fitness
        if nn.fitness > max :
            max = nn.fitness
            maxNeuralNet = nn

    avg/=pop.popCount
    print("Generation : %3d |  Avg Fitness : %4.0f  |  Max Fitness : %4.0f  " % (gen+1, avg, max) )
    MAXFIT.append(max) 
    AVGFIT.append(avg)
    pop.newGen()
        
env.close()

plt.figure();
plt.plot(range(MAX_GENERATIONS), AVGFIT)
plt.plot(range(MAX_GENERATIONS), MAXFIT)
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend(['Mean fitness', 'Max fitness'])
plt.show()