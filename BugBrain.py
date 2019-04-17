#!usr/bin/python
import matplotlib.pyplot as plt
import networkx as nx
import random
import math
# bias and weight should be in [-1, 1]
# decay should be in [0, 1]
# the smaller decay is, the faster the decay are.
G = nx.DiGraph()
plt.ion()
fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1)
types = {'Step': 0, 'Linear': 1, 'Sigmoid': 2, 'Tanh': 3, 'Relu': 4}
ACTIVE_MIN = 0.9
UNUSED_TIME_RATE = 200


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def tanh(x):
    expx = math.exp(x)
    exp_x = math.exp(-x)
    return (expx - exp_x) / (expx + exp_x)


def safe_param(x):
    return 1 if x > 1 else (-1 if x < -1 else x)


class Brain:
    def __init__(self, param_num=0):
        self.param_num = param_num
        self.neurons = []
        self.feel = []
        self.reward = InputNode(sid='r')

    def work(self):
        for neuron in reversed(self.neurons):
            neuron.count()
        for neuron in self.neurons:
            if neuron.stable >= 0.9 and neuron.good <= 0.1:
                neuron.bias = 2 * random.random() - 1
                neuron.good = 1
                neuron.stable = 0

    def draw(self):
        G.clear()
        ax.cla()
        node_colors = []
        edge_colors = []
        for neuron in self.neurons:
            G.add_node(neuron.sid)
            node_colors.append((neuron.value + 1) / 2)
        for in_node in self.feel:
            G.add_node(in_node.sid)
            node_colors.append((in_node.value + 1) / 2)
        for reward in [self.reward]:
            G.add_node(reward.sid)
            node_colors.append((reward.value + 1) / 2)
        for neuron in self.neurons:
            for synapse in neuron.synapses:
                G.add_edge(synapse.neu.sid, neuron.sid)
                edge_colors.append((synapse.value + 1) / 2)
        nx.draw_shell(G, ax=ax, with_labels=True,
                      node_color=node_colors,
                      edge_color=edge_colors,
                      cmap=plt.get_cmap('coolwarm'),
                      edge_cmap=plt.get_cmap('coolwarm'),
                      vmin=0, vmax=1)
        plt.pause(0.0001)

    def mutation(self, running=False, effect=1):
        if running:
            mutation_rate = {'param': 0, 'struct': 0.05}
        else:
            mutation_rate = {'param': 0.65, 'struct': 0.1}
        if not running:
            param = []
            for neuron in self.neurons:
                param.append(neuron.bias)
                for synapse in neuron.synapses:
                    param.append(synapse.weight)
                    param.append(synapse.decay)
            new_param = list(param)
            for j in range(len(param)):
                if random.random() <= mutation_rate['param']:
                    new_param[j] = param[j] + (random.random() - 0.5) * tanh(effect * 5)
            i_param = 0
            for j in range(len(self.neurons)):
                self.neurons[j].bias = safe_param(new_param[i_param])
                i_param += 1
                for k in range(len(self.neurons[j].synapses)):
                    self.neurons[j].synapses[k].weight = safe_param(new_param[i_param])
                    i_param += 1
                    self.neurons[j].synapses[k].decay = safe_param(new_param[i_param])
                    i_param += 1
        if random.random() <= mutation_rate['struct']:
            degree = len(self.neurons)
            i = random.randint(0, degree-1)
            j = random.randint(0, self.param_num + degree)
            exist = False
            for synapse in self.neurons[i].synapses:
                if synapse.sid == j:
                    exist = True
                    break
                elif synapse.sid > j:
                    break
            if not exist:
                if j < self.param_num:
                    neu = self.feel[j]
                elif j == self.param_num:
                    neu = self.reward
                else:
                    neu = self.neurons[j - self.param_num - 1]
                self.neurons[i].synapses.append(Synapse(neu,
                                                        sid=j,
                                                        weight=2*random.random()-1,
                                                        decay=random.random()))
                self.neurons[i].synapses.sort(key=lambda x: x.sid)


class Synapse:
    def __init__(self, neu, sid=0, weight=1.0, decay=0, learn=True):
        self.neu = neu
        self.sid = sid
        self.weight = weight
        self.decay = decay
        self.learn = learn
        self.stable = 0
        self.using_freq = 1
        self.good = 1
        self.active = False
        self.decayed = False
        self.value = 0

    def count(self):
        weighted_value = self.neu.value * self.weight
        if math.fabs(weighted_value) >= ACTIVE_MIN:
            self.using_freq = (self.using_freq * 8 + 1) / 9
        else:
            self.using_freq = self.using_freq * 8 / 9
        self.good = (self.good * 8 + self.using_freq) / 9
        if self.learn:
            if random.random() <= (1 - self.good) * (1 - self.stable):
                self.stable = (self.stable * 20 + 1) / 21
                self.weight = safe_param(self.weight + (random.random() - 0.5) * tanh((1 - self.good) * 5))
                self.decay = safe_param(self.decay + (random.random() - 0.5) * tanh((1 - self.good) * 5))
            else:
                if random.random() <= self.good ** 2:
                    self.stable = (self.stable * 20 + 1) / 21
        self.decay = math.fabs(self.decay)
        if self.decay == 0 or self.decay == 1:
            self.value = weighted_value
            if self.learn and random.random() > self.good:
                self.value = 0
        else:
            if self.active:
                self.value = self.value * pow(self.decay, 0.3)
                if math.fabs(self.value) > math.fabs(weighted_value):
                    self.value = weighted_value
                if math.fabs(self.value) <= 0.01:
                    self.active = False
                    self.decayed = True
                    self.value = 0
            else:
                if math.fabs(weighted_value) >= ACTIVE_MIN:
                    if not self.decayed:
                        self.active = True
                        self.value = weighted_value
                else:
                    self.decayed = False
                    self.value = weighted_value
                if self.learn and random.random() > self.good:
                    self.active = False
                    self.value = 0
        return self.value


class InputNode:
    def __init__(self, sid=0):
        self.sid = sid
        self.value = 0


class Neuron:
    def __init__(self, the_type, sid=0, bias=0.5, learn=True):
        self.__type = types[the_type]
        self.sid = sid
        self.bias = bias
        self.synapses = []
        self.learn = learn
        self.stable = 0
        self.using_freq = 0
        self.good = 1
        self.value = 0

    def count(self):
        s_sum = 0
        for s in self.synapses:
            s_sum += s.count()
        s_sum -= self.bias
        if s_sum >= 0:
            self.using_freq = (self.using_freq * 8 + 1) / 9
        else:
            self.using_freq = self.using_freq * 8 / 9
        self.good = (self.good * 8 + self.using_freq) / 9
        if self.learn:
            if random.random() <= (1 - self.good) * (1 - self.stable):
                self.stable = (self.stable * 8 + 1) / 9
                self.bias = safe_param(self.bias + (random.random() - 0.5) * tanh((1 - self.good) * 5))
            else:
                if random.random() <= self.good ** 2:
                    self.stable = (self.stable * 20 + 1) / 21
        if self.__type == types['Step']:
            if s_sum >= 0:
                self.value = 1
            else:
                self.value = 0
        elif self.__type == types['Linear']:
            self.value = s_sum
        elif self.__type == types['Sigmoid']:
            self.value = sigmoid(s_sum)
        elif self.__type == types['Tanh']:
            self.value = tanh(s_sum)
        elif self.__type == types['Relu']:
            if s_sum > 0:
                self.value = s_sum
            else:
                self.value = 0
        if self.learn and random.random() > self.good:
            self.value = 0
        # below may be no use
        for synapse in self.synapses:
            if synapse.stable >= 0.9 and (abs(synapse.weight) <= 0.1 or synapse.good <= 0.1):
                self.synapses.remove(synapse)
