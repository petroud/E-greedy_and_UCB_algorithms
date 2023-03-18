import numpy as np
import matplotlib.pyplot as plt


class KStochasticBandits:
    def __init__(self, k):
        self.k = k
        self.a = np.random.rand(k)
        self.b = np.random.rand(k)
        self.mu = (self.a + self.b) / 2
        
    def pull(self, i):
        return np.random.uniform(self.a[i], self.b[i])


class EGreedy:
    def __init__(self, k, epsilon):
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        
    def select_bandit(self):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.k)
        else:
            return np.argmax(self.Q)
        
    def update(self, i, r):
        self.N[i] += 1
        self.Q[i] += (r - self.Q[i]) / self.N[i]


class UCB:
    def __init__(self, k, c):
        self.k = k
        self.c = c
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.t = 0
        
    def select_bandit(self):
        if self.t < self.k:
            return self.t
        else:
            ucb = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
            return np.argmax(ucb)
        
    def update(self, i, r):
        self.N[i] += 1
        self.Q[i] += (r - self.Q[i]) / self.N[i]
        self.t += 1


np.random.seed(123)

# create the bandits environment
bandits = KStochasticBandits(k=10)

# create the ε-Greedy and UCB agents
e_greedy_agent = EGreedy(k=10, epsilon=0.1)
ucb_agent = UCB(k=10, c=2)

# run the simulation for T steps
T = 1000
for t in range(T):
    # ε-Greedy action selection
    i_e = e_greedy_agent.select_bandit()
    r_e = bandits.pull(i_e)
    e_greedy_agent.update(i_e, r_e)
    
    # UCB action selection
    i_u = ucb_agent.select_bandit()
    r_u = bandits.pull(i_u)
    ucb_agent.update(i_u, r_u)

# calculate the regret for each time step
regret_e = np.zeros(T)
regret_u = np.zeros(T)
for t in range(1, T):
    max_mu = np.max(bandits.mu)
    regret_e[t] = regret_e[t-1] + max_mu - bandits.pull(np.argmax(e_greedy_agent.Q))
    regret_u[t] = regret_u[t-1] + max_mu - bandits.pull(np.argmax(ucb_agent.Q))

# plot the regrets
plt.plot(regret_e, label='ε-Greedy')
plt.plot(regret_u, label='UCB')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Regret')
plt.show()