#!usr/bin/python
import BugBrain as BB
import random
import gym
env = gym.make("CartPole-v1")
observation = env.reset()
param_num = len(observation)
rend = False


class worm:
    def __init__(self, number):
        self.number = number
        self.brain = BB.Brain()
        self.feel = []
        for _ in range(param_num):
            self.feel.append(BB.InputNode())
        self.effect = 0
        self.brain.neurons.append(BB.Neuron('Step', bias=2*random.random()-1))
        for i in range(param_num):
            self.brain.neurons[0].synapses.append(BB.Synapse(self.feel[i], weight=2*random.random()-1))

    def work(self):
        observation = env.reset()
        effect = 0
        for t in range(200):
            if rend:
                env.render()
            for i in range(param_num):
                self.feel[i].value = observation[i]
            self.brain.work()
            action = self.brain.neurons[0].value
            observation, reward, done, info = env.step(action)
            effect += reward
            if done:
                self.effect = 1.0 / (t + 1)
                break


class Teacher:
    def __init__(self, max_num, keep_num):
        self.worms = []
        self.max_num = max_num
        self.keep_num = keep_num
        self.generation = 0
        for i in range(self.max_num):
            self.worms.append(worm(i))

    def show(self):
        for i in range(self.keep_num):
            print(round(self.worms[i].number, 5), end=' ')
        print()
        for i in range(self.keep_num):
            print(round(self.worms[i].effect, 5), end=' ')
        print()

    def generate(self):
        self.generation += 1
        for i in range(self.keep_num, self.max_num):
            param = self.worms[i % self.keep_num].brain.parameter()
            new_param = list(param)
            for j in range(len(param)):
                new_param[j] = param[j] + random.random() - 0.5
            self.worms[i].brain.updateParam(new_param)

    def work(self):
        for i in range(self.max_num):
            self.worms[i].work()
        self.worms.sort(key=lambda x: x.effect, reverse=False)


if __name__ == "__main__":
    teacher = Teacher(max_num=32, keep_num=8)
    teacher.work()
    teacher.show()
    for i in range(10):
        if i > 8:
            rend = True
        teacher.generate()
        teacher.work()
        teacher.show()
env.close()
