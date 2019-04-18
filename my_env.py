#!usr/bin/python
import matplotlib.pyplot as plt
fig = plt.figure(2)
ax = fig.add_subplot(1, 1, 1)
line = ax.plot([0, 100], [0, 1], '-g')[0]
line2 = ax.plot([0, 100], [0, 1], '-y')[0]
plt.ion()


class Remember:
    def __init__(self):
        self.reset()

    def reset(self):
        self.time = 0
        self.X = []
        self.Y = []
        self.Y2 = []
        return [0]

    def render(self):
        self.X.append(self.time)
        self.Y.append(self.action)
        self.Y2.append(self.observation[0])
        line.set_xdata(self.X)
        line2.set_xdata(self.X)
        line.set_ydata(self.Y)
        line2.set_ydata(self.Y2)
        plt.pause(0.0001)

    def step(self, action):
        self.time += 1
        self.action = action[0]
        self.observation = [1.0*(self.time >= 20 and self.time <= 40)]
        reward = abs(action[0] - (self.time >= 60 and self.time <= 80))
        done = self.time > 100
        return self.observation, reward, done, {}

    def close(self):
        pass


class And_or_Or:
    def __init__(self):
        self.reset()

    def reset(self):
        self.time = 0
        self.X = []
        self.Y = []
        self.Y2 = []
        return [0, 0]

    def render(self):
        self.X.append(self.time)
        self.Y.append(self.action)
        self.Y2.append(self.expect)
        line.set_xdata(self.X)
        line2.set_xdata(self.X)
        line.set_ydata(self.Y)
        line2.set_ydata(self.Y2)
        plt.pause(0.0001)

    def step(self, action):
        self.time += 1
        self.action = action[0]
        a = 1.0 * ((self.time // 4) % 2)
        b = 1.0 * ((self.time // 4) % 4 > 1)
        self.observation = [a, b]
        if self.time > 50:
            self.expect = 1.0 * (a and b)
        else:
            self.expect = 1.0 * (a or b)
        reward = abs(action[0] - self.expect)
        done = self.time > 100
        return self.observation, reward, done, {}

    def close(self):
        pass


games = {"Remember": Remember, "And_or_Or": And_or_Or}


def make(env_name):
    return games[env_name]()


if __name__ == "__main__":
    env = make("Remember")
    for _ in range(5):
        env.reset()
        for i in range(100):
            env.step([0])
            env.render()
