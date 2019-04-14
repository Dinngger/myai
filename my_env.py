#!usr/bin/python
import matplotlib.pyplot as plt
fig = plt.figure(2)
ax = fig.add_subplot(1, 1, 1)
line = ax.plot([0, 100], [0, 1], '-g')[0]
plt.ion()


class Remember:
    def __init__(self):
        self.reset()

    def reset(self):
        self.time = 0
        self.X = []
        self.Y = []
        return [0]

    def render(self):
        self.X.append(self.time)
        self.Y.append(self.action)
        line.set_xdata(self.X)
        line.set_ydata(self.Y)
        plt.pause(0.0001)

    def step(self, action):
        self.time += 1
        self.action = action[0]
        self.observation = [1.0*(self.time >= 20 and self.time <= 40)]
        reward = 1.0*(action[0] ^ (self.time >= 60 and self.time <= 80))
        done = self.time > 100
        return self.observation, reward, done, {}

    def close(self):
        pass


games = {"Remember": Remember}


def make(env_name):
    return games[env_name]()


if __name__ == "__main__":
    env = make("Remember")
    for _ in range(5):
        env.reset()
        for i in range(100):
            env.step([0])
            env.render()
