import gym
import numpy as np

amap=['SFFF','HFFF','FFFF','FFFG']
env = gym.make('FrozenLake-v0',desc=amap)
env.seed(741684)
np.random.seed(741684)

gym.envs.toy_text.frozen_lake.FrozenLakeEnv().unwrapped

#max_steps = 100


gamma = 1
alpha = 0.25
epsilon = 0.29
n_episodes = 14697

Q = np.zeros((env.observation_space.n, env.action_space.n))

def select_a_with_epsilon_greedy(curr_s, q_value, epsilon=0.1):
    a = np.argmax(q_value[curr_s, :])
    if np.random.rand() < epsilon:
        a = np.random.randint(q_value.shape[1])
    return a

for episode in range(n_episodes):

        # Reset a cumulative reward for this episode
        cumu_r = 0

        # Start a new episode and sample the initial state
        curr_s = env.reset()

        # Select the first acti
        curr_a= select_a_with_epsilon_greedy(curr_s, Q, epsilon=epsilon)
  
        while 1:
            next_s, r, done, info = env.step(curr_a)
            cumu_r = r + gamma * cumu_r
            next_a = select_a_with_epsilon_greedy(next_s, Q, epsilon=epsilon)
            delta = r + gamma * Q[next_s, next_a] - Q[curr_s, curr_a]
            Q[curr_s, curr_a] += alpha * delta
    
            curr_s = next_s
            curr_a = next_a
            if done:
                break
            
mapping=["<","V",">","^"]
result=[]
for row in range(len(Q)):
    result.append(mapping[list(Q[row]).index(max(Q[row]))])
print(result)
