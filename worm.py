#!usr/bin/python
import BugBrain as BB
import pickle
import random
import math
import copy
# import gym
import my_env
env_name = 'Remember'
env = my_env.make(env_name)  # gym.make("Pong-ram-v0")
observation = env.reset()
env.step([0])
param_num = len(observation)
rend = False
single_rend = False
GAUSSIAN_RATE = 4


def gaussian(x):
    return math.exp(-pow(x/param_num, 2)) * 0.9


class Worm:
    def __init__(self, sid, degree=3):
        self.sid = sid
        self.brain = BB.Brain(input_dimension=param_num, output_dimension=1)
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
                    self.brain.neurons[i].synapses.append(BB.Synapse(self.brain.neurons[j], sid=j+param_num))
            for j in range(degree // 2, degree):
                if random.random() <= gaussian((j - i - 1) / GAUSSIAN_RATE):
                    self.brain.neurons[i].synapses.append(BB.Synapse(self.brain.neurons[j], sid=j+param_num))
            for j in range(param_num):
                if random.random() <= gaussian((j + degree - i - 1) / GAUSSIAN_RATE):
                    self.brain.neurons[i].synapses.append(BB.Synapse(self.brain.feel[j], sid=j))
            for synapse in self.brain.neurons[i].synapses:
                synapse.weight = 2*random.random()-1
                synapse.decay = random.random()
            self.brain.neurons[i].synapses.sort(key=lambda x: x.sid)

    def mutation(self):
        self.brain.mutation()

    def work(self):
        one_work_time = 100
        for _ in range(2):
            effect = 0
            observation = env.reset()
            for t in range(one_work_time):
                for i in range(param_num):
                    self.brain.feel[i].value = BB.tanh(observation[i])
                action = self.brain.work()
                observation, reward, done, info = env.step(action)
                if self.brain.is_good and (rend and single_rend):
                    env.render()
                    if not t % 10:
                        self.brain.draw()
                effect += reward
                if done:
                    break
            # self.effect shoule better be in about 1
            self.effect = effect / one_work_time
            self.brain.update_effect(self.effect)
            if self.brain.effect == 0 or self.brain.not_update >= 5:
                self.brain.show_good()
            else:
                self.brain.mutation(running=True)


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
        '''
        print('new generation:')
        for i in range(self.max_num):
            print(round(self.worms[i].brain.effect, 5), end=' ')
        print()
        for i in range(self.max_num):
            print(round(self.worms[i].effect, 5), end=' ')
        print()

    def generate(self):
        self.generation += 1
        self.biodiverse = 0
        first_effect = 0
        if self.worms[0].brain.effect != 0:
            for i in range(self.keep_num):
                if self.worms[i].brain.effect != first_effect:
                    first_effect = self.worms[i].brain.effect
                    self.biodiverse = 0
                else:
                    self.biodiverse += 1
                    if self.biodiverse > self.keep_num // 2:
                        break
            if self.biodiverse > self.keep_num // 2:
                for i in range(self.max_num-1, 1, -1):
                    if self.worms[i].brain.effect == first_effect:
                        self.worms.remove(self.worms[i])
        for i in range(0, self.max_num):
            if i >= len(self.worms):
                if len(self.worms) >= self.keep_num:
                    self.worms.append(copy.deepcopy(self.worms[i % self.keep_num]))
                else:
                    self.worms.append(copy.deepcopy(self.worms[i % len(self.worms)]))
            else:
                self.worms[i] = copy.deepcopy(self.worms[i % self.keep_num])
            self.worms[i].mutation()

    def work(self):
        global single_rend
        for i in range(self.max_num):
            single_rend = i <= 1  # self.keep_num
            self.worms[i].work()
            if i == 0:
                with open('./brains/brain_{}'.format(env_name), 'wb') as f:
                    pickle.dump(self.worms[i], f)
        self.worms.sort(key=lambda x: x.brain.effect, reverse=False)


if __name__ == "__main__":
    teacher = Teacher(max_num=128, keep_num=32, load=False)
    teacher.work()
    teacher.show()
    for i in range(150):
        if teacher.keep_num > 32 and not i % 4:
            teacher.keep_num = teacher.keep_num // 2
        rend = i != 0 and not i % 5
        teacher.generate()
        teacher.work()
        teacher.show()
env.close()
