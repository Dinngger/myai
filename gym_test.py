import gym
env = gym.make("Pendulum-v0")
observation = env.reset()
for t in range(100):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(observation)
    print(action)
env.close()
