#!usr/bin/python
import BugBrain as BB
import random
import math
import copy
import gym
env = gym.make("Pendulum-v0")
observation = env.reset()
param_num = len(observation)
rend = True
single_rend = False
GAUSSIAN_RATE = 4


def gaussian(x):
    return math.exp(-x*x) / 2


class worm:
    def __init__(self, number):
        self.number = number
        self.brain = BB.Brain()
        self.feel = []
        for i in range(param_num):
            self.feel.append(BB.InputNode(sid='i{}'.format(i)))
        self.reward = BB.InputNode(sid='r')
        self.effect = 0
        self.brain_generator(3)

    def brain_generator(self, degree):
        for i in range(degree):
            self.brain.neurons.append(BB.Neuron('Tanh', sid=i, bias=2*random.random()-1))
        for i in range(degree):
            for j in range(degree // 2):
                if random.random() <= gaussian(j - i - 2):
                    self.brain.neurons[i].synapses.append(BB.Synapse(self.brain.neurons[j], sid=j+param_num+1))
            if random.random() <= gaussian(degree // 2 - i - 2):
                self.brain.neurons[i].synapses.append(BB.Synapse(self.reward, sid=param_num))
            for j in range(degree // 2, degree):
                if random.random() <= gaussian((j - i - 1) / GAUSSIAN_RATE):
                    self.brain.neurons[i].synapses.append(BB.Synapse(self.brain.neurons[j], sid=j+param_num+1))
            for j in range(param_num):
                if random.random() <= gaussian((j + degree - i - 1) / GAUSSIAN_RATE):
                    self.brain.neurons[i].synapses.append(BB.Synapse(self.feel[j], sid=j))
            for synapse in self.brain.neurons[i].synapses:
                synapse.weight = 2*random.random()-1
                synapse.decay = pow(random.random(), 4)
            self.brain.neurons[i].synapses.sort(key=lambda x: x.sid)

    def work(self):
        effect = 0
        work_times = 3
        for _ in range(work_times):
            single_effect = 0
            observation = env.reset()
            for t in range(300):
                if rend and single_rend:
                    env.render()
                for i in range(param_num):
                    self.feel[i].value = observation[i]
                self.brain.work()
                if rend and single_rend and not t % 20:
                    self.brain.draw(self.feel, self.reward)
                action = [self.brain.neurons[0].value]
                observation, reward, done, info = env.step(action)
                single_effect -= reward
                if done:
                    effect += single_effect / (t + 1)
                    break
        self.effect = effect / work_times


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
            self.worms[i] = copy.deepcopy(self.worms[i % self.keep_num])
            param = self.worms[i].brain.parameter()
            new_param = list(param)
            for j in range(len(param)):
                if random.random() >= 0.9:
                    new_param[j] = param[j] + random.random() - 0.5
            self.worms[i].brain.updateParam(new_param)

    def work(self):
        global single_rend
        for i in range(self.max_num):
            single_rend = i < self.keep_num
            self.worms[i].work()
        self.worms.sort(key=lambda x: x.effect, reverse=False)


if __name__ == "__main__":
    teacher = Teacher(max_num=32, keep_num=8)
    teacher.work()
    teacher.show()
    for i in range(50):
        if i >= 1:
            rend = True
        teacher.generate()
        teacher.work()
        teacher.show()
env.close()
