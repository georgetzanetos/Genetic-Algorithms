from matplotlib import pyplot as plt
from CTRNN import CTRNN
from scipy.sparse import csr_matrix
import gym
import numpy as np

# added unpacking of genome:
class CTRNN_agent(object):
    
    """ Continuous Time Recurrent Neural Network agent. """
    
    n_observations = 4;
    n_actions = 1;
    
    def __init__(self, network_size, genome = [], weights=[], taus = [], gains = [], biases = []):
        
        self.network_size = network_size;
        if(self.network_size < self.n_observations + self.n_actions):
            self.network_size = self.n_observations + self.n_actions;
        self.cns = CTRNN(self.network_size, step_size=0.1) 
        
        if(len(genome) == self.network_size*self.network_size+3*self.network_size):
            # Get the network parameters from the genome:
            weight_range = 3
            ind = self.network_size*self.network_size
            w = weight_range * (2.0 * (genome[:ind] - 0.5))
            weights = np.reshape(w, [self.network_size, self.network_size])
            biases = weight_range * (2.0 * (genome[ind:ind+self.network_size] - 0.5))
            ind += self.network_size
            taus = 0.9 * genome[ind:ind+self.network_size] + 0.05
            ind += self.network_size
            gains = 2.0 * (genome[ind:ind+self.network_size]-0.5)
        
        if(len(weights) > 0):
            # weights must be a matrix size: network_size x network_size
            self.cns.weights = csr_matrix(weights)
        if(len(biases) > 0):
            self.cns.biases = biases
        if(len(taus) > 0):
            self.cns.taus = taus
        if(len(gains) > 0):
            self.gains = gains
    
    def act(self, observation, reward, done):
        external_inputs = np.asarray([0.0]*self.network_size)
        external_inputs[0:self.n_observations] = observation
        self.cns.euler_step(external_inputs)
        output = (self.cns.outputs[-self.n_actions:])
        return output

def run_cartpole(agent, simulation_seed=0, n_episodes=1, env=gym.make('CartPole-v1'), max_steps = 1000, graphics=False):

    env.seed(simulation_seed)

    reward = 0
    cumulative_reward = 0
    done = False
    step = 0

    for i in range(n_episodes):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            cumulative_reward += reward
            step += 1
            if(step >= max_steps):
                done = True
            if(graphics):
                env.render()
            if done:
                break

    # env.close()    
    
    return cumulative_reward;

def evaluate(genome, seed = 0, graphics = False, original_reward=True):
    # create the phenotype from the genotype:
    n_neurons  = 10
    weights = np.zeros([n_neurons, n_neurons])
    taus = np.asarray([0.1]*n_neurons)
    gains = np.ones([n_neurons,])
    biases = np.zeros([n_neurons,])
    agent = CTRNN_agent(n_neurons, weights=weights, taus = taus, gains = gains, biases = biases)
    reward = run_cartpole(agent, simulation_seed=0, env=gym.make('CartPole-v0'), graphics=True)

    agent = CTRNN_agent(n_neurons, genome=genome)
    
    # run the agent:
    if(original_reward):
        reward = run_cartpole(agent, simulation_seed=seed, graphics=graphics)
    else:
        reward = run_cartpole(agent, env=gym.make('CartPole-v0'), simulation_seed=seed, graphics=graphics)
    #print('Reward = ' + str(reward))
    return reward


def test_best(Best, original_reward=True):    
    n_tests = 30
    fit = np.zeros([n_tests,])
    for t in range(n_tests):
        fit[t] = evaluate(Best, seed = 100+t, graphics=False, original_reward=True)
    
    plt.figure()
    plt.boxplot(fit)
    plt.ylabel('Fitness')
    plt.xticks([1], ['Fitness best individual'])
    
# Parameters CTRNN:
network_size = 10
genome_size = (network_size+3)*network_size

# Evolutionary algorithm:
n_individuals = 1
n_generations = 5
p_mut = 0.05
n_best = 3

np.random.seed(0) # 0-5 do not work
original_reward = False
Population = np.random.rand(n_individuals, genome_size)
Reward = np.zeros([n_individuals,])
max_fitness = np.zeros([n_generations,])
mean_fitness = np.zeros([n_generations,])
Best = []
fitness_best = []

for g in range(n_generations):
    for i in range (n_individuals):
        Reward[i] = evaluate(Population[i, :], original_reward=original_reward)


# for g in range(n_generations):
    
#     # evaluate:
#     for i in range(n_individuals):
#         Reward[i] = evaluate(Population[i, :], original_reward=original_reward)
#     mean_fitness[g] = np.mean(Reward)
#     max_fitness[g] = np.max(Reward)
#     print('Generation {}, mean = {} max = {}'.format(g, mean_fitness[g], max_fitness[g]))
#     # select:
#     inds = np.argsort(Reward)
#     inds = inds[-n_best:]
#     if(len(Best) == 0 or Reward[-1] > fitness_best):
#         Best = Population[inds[-1], :] 
#         fitness_best = Reward[-1]
#     # vary:
#     NewPopulation = np.zeros([n_individuals, genome_size])
#     for i in range(n_individuals):
#         ind = inds[i % n_best]
#         NewPopulation[i,:] = Population[ind, :]
#         for gene in range(genome_size):
#             if(np.random.rand() <= p_mut):
#                 NewPopulation[i,gene] = np.random.rand()
#     Population = NewPopulation

# print('Best fitness ' + str(fitness_best))
# print('Genome = ')
# for gene in range(len(Best)):
#     if(gene == 0):
#         print('[' + str(Best[gene]) + ', ', end='');
#     elif(gene == len(Best)-1):
#         print(str(Best[gene]) + ']');
#     else:
#         print(str(Best[gene]) + ', ', end='');

# plt.figure();
# plt.plot(range(n_generations), mean_fitness)
# plt.plot(range(n_generations), max_fitness)
# plt.xlabel('Generations')
# plt.ylabel('Fitness')
# plt.legend(['Mean fitness', 'Max fitness'])

# evaluate(Best, graphics=True)
# test_best(Best)