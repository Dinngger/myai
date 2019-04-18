#!usr/bin/python
import BugBrain as BB
import pickle
import random
import math
import copy
# import gym
import my_env
env_name = 'And_or_Or'
env = my_env.make(env_name)  # gym.make("Pong-ram-v0")
observation = env.reset()
env.step([0])
param_num = len(observation)
rend = True
single_rend = False
GAUSSIAN_RATE = 4


def gaussian(x):
    return math.exp(-pow(x/param_num, 2)) * 0.9


class Worm:
    def __init__(self, sid, degree=3):
        self.sid = sid
        self.brain = BB.Brain(param_num=param_num)
        for i in range(param_num):
            self.brain.feel.append(BB.InputNode(sid='i{}'.format(i)))
        self.effect = 0
        self.brain_generator(degree)

    def brain_generator(self, degree):
        for i in range(degree):
            self.brain.neurons.append(BB.Neuron('Step', sid=i, bias=2*random.random()-1))
        for i in range(degree):
            for j in range(degree // 2):
                if random.random() <= gaussian(j - i - 2):
                    self.brain.neurons[i].synapses.append(BB.Synapse(self.brain.neurons[j], sid=j+param_num+1))
            if random.random() <= gaussian(degree // 2 - i - 2):
                self.brain.neurons[i].synapses.append(BB.Synapse(self.brain.reward, sid=param_num))
            for j in range(degree // 2, degree):
                if random.random() <= gaussian((j - i - 1) / GAUSSIAN_RATE):
                    self.brain.neurons[i].synapses.append(BB.Synapse(self.brain.neurons[j], sid=j+param_num+1))
            for j in range(param_num):
                if random.random() <= gaussian((j + degree - i - 1) / GAUSSIAN_RATE):
                    self.brain.neurons[i].synapses.append(BB.Synapse(self.brain.feel[j], sid=j))
            for synapse in self.brain.neurons[i].synapses:
                synapse.weight = 2*random.random()-1
                synapse.decay = random.random()
            self.brain.neurons[i].synapses.sort(key=lambda x: x.sid)

    def mutation(self):
        self.brain.mutation(effect=self.effect)

    def work(self):
        work_times = 10
        one_work_time = 100
        for _ in range(work_times):
            observation = env.reset()
            effect = 0
            for t in range(one_work_time):
                for i in range(param_num):
                    self.brain.feel[i].value = BB.tanh(observation[i])
                self.brain.work()
                action = [self.brain.neurons[0].value]
                observation, reward, done, info = env.step(action)
                self.brain.reward.value = reward
                if rend and single_rend:
                    env.render()
                    if not t % 10:
                        self.brain.draw()
                if t >= 0:
                    effect += reward
                if done:
                    break
                self.brain.mutation(running=True)
            # self.effect shoule better be in about 1
            self.effect = effect / one_work_time


class Teacher:
    def __init__(self, max_num, keep_num, load=False):
        self.worms = []
        self.max_num = max_num
        self.keep_num = keep_num
        self.generation = 0
        if load:
            with open('./brains/{}'.format(env_name), 'rb') as f:
                load_worm = pickle.load(f)
        for i in range(self.max_num):
            if load:
                self.worms.append(load_worm)
            else:
                self.worms.append(Worm(i))
        if load:
            self.generate()
            for i in range(self.max_num):
                self.worms[i].sid = i

    def show(self):
        '''
        for i in range(self.max_num):
            print(round(self.worms[i].sid, 5), end=' ')
        print()
        '''
        for i in range(self.max_num):
            print(round(self.worms[i].effect, 5), end=' ')
        print()

    def generate(self):
        self.generation += 1
        self.biodiverse = 0
        first_effect = 0
        for i in range(self.keep_num):
            if self.worms[i].effect != first_effect:
                first_effect = self.worms[i].effect
                self.biodiverse = 0
            else:
                self.biodiverse += 1
                if self.biodiverse > self.keep_num / 2:
                    break
        if self.biodiverse > self.keep_num / 2:
            for i in range(self.max_num-1, -1, -1):
                if self.worms[i].effect == first_effect:
                    self.worms.remove(self.worms[i])
        stage = 1
        for j in range(stage):
            for i in range(self.max_num//stage*j + self.keep_num//stage, (j+1)*self.max_num//stage):
                if i >= len(self.worms):
                    self.worms.append(copy.deepcopy(self.worms[i % self.keep_num + j*self.max_num//stage]))
                else:
                    self.worms[i] = copy.deepcopy(self.worms[i % self.keep_num + j*self.max_num//stage])
                self.worms[i].mutation()

    def work(self):
        global single_rend
        for i in range(self.max_num):
            single_rend = i == 0  # self.keep_num
            self.worms[i].work()
            if i == 0:
                with open('./brains/brain_{}'.format(env_name), 'wb') as f:
                    pickle.dump(self.worms[i], f)
        self.worms.sort(key=lambda x: x.effect, reverse=False)


if __name__ == "__main__":
    teacher = Teacher(max_num=64, keep_num=16, load=False)
    teacher.work()
    teacher.show()
    for i in range(200):
        if teacher.keep_num > 32 and not i % 4:
            teacher.keep_num = teacher.keep_num // 2
        if i != 0 and not i % 100:
            rend = True
        teacher.generate()
        teacher.work()
        teacher.show()
env.close()
