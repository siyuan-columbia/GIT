import gym
import numpy as np

amap=['SFF','HFF','HHG']
env = gym.make('FrozenLake-v0',desc=amap)
env.seed(455090)
np.random.seed(455090)

gym.envs.toy_text.frozen_lake.FrozenLakeEnv().unwrapped

max_steps = 100


gamma = 0.96
alpha = 0.07
epsilon = 0.18
n_episodes = 45930

Q = np.zeros((env.observation_space.n, env.action_space.n))
    
def choose_action(state):
    action=0
    if np.random.randint(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def learn(state, state2, reward, action):
    predict = Q[state, action]
    target = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = Q[state, action] + alpha * (target - predict)

# Start
for episode in range(n_episodes):
    state = env.reset()
    t = 0
    
    while t < max_steps:
        env.render()

        action = choose_action(state)  

        state2, reward, done, info = env.step(action)  

        learn(state, state2, reward, action)

        state = state2

        t += 1
       
        if done:
            break

mapping=["<","V",">","^"]
result=[]
for row in range(len(Q)):
    result.append(mapping[list(Q[row]).index(max(Q[row]))])
    