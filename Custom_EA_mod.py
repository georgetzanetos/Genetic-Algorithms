import gym
from matplotlib import pyplot as plt
import random, bisect
import numpy as np

seed=10
# np.random.seed(seed)
# random.seed(seed)

#sigmoid activation function
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


class NN_agent :
    
    def __init__(self, n_nodes):     

        global seed

        self.weights = []
        self.biases = []

        self.fitness = 0
        self.nodes = n_nodes

        for i in range(len(n_nodes) - 1):
            np.random.seed(seed)
            self.weights.append(np.random.uniform(low=-1, high=1, size=(n_nodes[i], n_nodes[i+1])).tolist())
            seed+=1
            np.random.seed(seed)
            self.biases.append(np.random.uniform(low=-1, high=1, size=(n_nodes[i+1])).tolist())
            seed+=1

    #evaluation
    def getOutput(self, input):

        output = input
        for i in range(len(self.nodes)-1):
            output = np.reshape(np.matmul(output, self.weights[i]) + self.biases[i], (self.nodes[i+1]))
        return np.argmax(sigmoid(output))

class Population :

    def __init__(self, n_individuals, p_mut, n_nodes):
        
        self.population = [NN_agent(n_nodes) for i in range(n_individuals)]
        self.n_nodes = n_nodes
        self.pop_size = n_individuals
        self.m_rate = p_mut
        
    #variation
    def createOffspring(self, parent1, parent2):

        global seed
        
        offspring = NN_agent(self.n_nodes)

        for i in range(len(offspring.weights)):
            for j in range(len(offspring.weights[i])):
                for k in range(len(offspring.weights[i][j])):
                    random.seed(seed)
                    if random.random() < self.m_rate:
                        seed+=1
                        random.seed(seed)
                        offspring.weights[i][j][k] = random.uniform(-1, 1)
                        seed+=1
                    else:
                        offspring.weights[i][j][k] = (parent1.weights[i][j][k] + parent2.weights[i][j][k])/2.0
                        seed+=1

        for i in range(len(offspring.biases)):
            for j in range(len(offspring.biases[i])):
                random.seed(seed)
                if random.random() < self.m_rate:
                    seed+=1
                    random.seed(seed)
                    offspring.biases[i][j] = random.uniform(-1, 1)
                    seed+=1
                else:
                    offspring.biases[i][j] = (parent1.biases[i][j] + parent2.biases[i][j])/2.0
                    seed+=1

        return offspring

    #selection
    def newGen(self, migration_ratio, elite):       

        global seed 

        total_fitness = [0]
        new_gen = []
        

        for i in range(len(self.population)):
            total_fitness.append(total_fitness[i]+self.population[i].fitness)

        for i in range(elite):
            new_gen.append(self.population[elite[i]])
            
        while(len(new_gen) < self.pop_size*migration_ratio):
            random.seed(seed)
            r1 = random.uniform(0, total_fitness[len(total_fitness)-1] )
            seed+=1
            random.seed(seed)
            r2 = random.uniform(0, total_fitness[len(total_fitness)-1] )
            seed+=1
            parent1 = self.population[bisect.bisect_right(total_fitness, r1)-1]
            parent2 = self.population[bisect.bisect_right(total_fitness, r2)-1]
            new_gen.append(self.createOffspring(parent1,parent2))

        while(len(new_gen) < self.pop_size):

            random.seed(seed)
            r1 = random.choice(elite)
            seed+=1
            parent1 = self.population[r1]

            random.seed(seed)
            r2 = random.choice(elite)
            seed+=1
            parent2 = self.population[r2]

            new_gen.append(self.createOffspring(parent1,parent2))

        self.population.clear()
        self.population = new_gen

STEPS = 500 
GENS = 30
POPULATION = 30
MUTATION = 0.01
MIGRATION = 0.5
ELITISM = 5

env = gym.make('CartPole-v1')

observation = env.reset()
dim_in = env.observation_space.shape[0]
dim_out = env.action_space.n
pop = Population(POPULATION, MUTATION, [dim_in, 8, 8, dim_out])

MAXFIT = []
AVGFIT = []

for gen in range(GENS):
    max = 0
    avg = 0
    TopGenome = None

    for genome in pop.population:
        Reward = 0
        
        for step in range(STEPS):
            env.render()
            action = genome.getOutput(observation)
            observation, reward, done, info = env.step(action)
            Reward += reward
            if done:
                observation = env.reset()
                break

        genome.fitness = Reward
        avg += genome.fitness

        if genome.fitness > max :
            max = genome.fitness
            TopGenome = genome
    
    idx = np.argsort(avg)
    elitidx = idx[-ELITISM:]

    avg/=pop.pop_size
    print("Generation : %3d |  Avg Fitness : %4.0f  |  Max Fitness : %4.0f  " % (gen+1, avg, max) )
    MAXFIT.append(max) 
    AVGFIT.append(avg)
    pop.newGen(MIGRATION,elitidx)
        
env.close()

plt.figure();
plt.plot(range(GENS), AVGFIT)
plt.plot(range(GENS), MAXFIT)
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend(['Mean fitness', 'Max fitness'])
plt.grid()
plt.show()

print(TopGenome)