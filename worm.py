#!usr/bin/python
import BugBrain as BB
import random
import math
import copy
import gym
env = gym.make("Pendulum-v0")
observation = env.reset()
param_num = len(observation)
rend = False
single_rend = False
GAUSSIAN_RATE = 4


def gaussian(x):
    return math.exp(-x*x/4) * 0.6


class worm:
    def __init__(self, number):
        self.number = number
        self.brain = BB.Brain()
        self.degree = 4
        self.feel = []
        for i in range(param_num):
            self.feel.append(BB.InputNode(sid='i{}'.format(i)))
        self.reward = BB.InputNode(sid='r')
        self.effect = 0
        self.brain_generator(self.degree)

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

    def mutation(self):
        param = self.brain.parameter()
        new_param = list(param)
        for j in range(len(param)):
            if random.random() >= 0.9:
                new_param[j] = param[j] + random.random() - 0.5
        self.brain.updateParam(new_param)
        i = random.randint(0, self.degree-1)
        j = random.randint(0, param_num + self.degree)
        for synapse in self.brain.neurons[i].synapses:
            if synapse.sid == j:
                self.brain.neurons[i].synapses.remove(synapse)
                break
            elif synapse.sid > j:
                if j < param_num:
                    neu = self.feel[j]
                elif j == param_num:
                    neu = self.reward
                else:
                    neu = self.brain.neurons[j - param_num - 1]
                self.brain.neurons[i].synapses.append(BB.Synapse(neu,
                                                                 sid=j,
                                                                 weight=2*random.random()-1,
                                                                 decay=pow(random.random(), 4)))
                self.brain.neurons[i].synapses.sort(key=lambda x: x.sid)
                break

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
                if rend and single_rend and not t % 100:
                    self.brain.draw(self.feel, self.reward)
                action = [self.brain.neurons[0].value * 3]
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
            self.worms[i].mutation()

    def work(self):
        global single_rend
        for i in range(self.max_num):
            single_rend = i < 1  # self.keep_num
            self.worms[i].work()
        self.worms.sort(key=lambda x: x.effect, reverse=False)


if __name__ == "__main__":
    teacher = Teacher(max_num=64, keep_num=16)
    teacher.work()
    teacher.show()
    for i in range(100):
        if i >= 10:
            rend = True
        teacher.generate()
        teacher.work()
        teacher.show()
env.close()
