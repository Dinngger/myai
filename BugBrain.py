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
MAX_TIME = 500


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def tanh(x):
    expx = math.exp(x)
    exp_x = math.exp(-x)
    return (expx - exp_x) / (expx + exp_x)


def safe_param(x):
    return 1 if x > 1 else (-1 if x < -1 else x)


class Brain:
    def __init__(self, input_dimension=0, output_dimension=0):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = []
        self.feel = []
        self.effect = 10
        self.not_update = 0
        self.is_good = False

    def work(self):
        for neuron in reversed(self.neurons):
            neuron.count()
        output = []
        for i in range(self.output_dimension):
            output.append(self.neurons[i].value)
        return output

    def update_effect(self, effect):
        if effect <= self.effect or self.not_update >= 5:
            self.not_update = 0
            self.effect = effect
            for neuron in self.neurons:
                neuron.good_bias = neuron.bias
                for synapse in neuron.synapses:
                    synapse.good_weight = synapse.weight
                    synapse.good_decay = synapse.decay
        else:
            self.not_update += 1

    def show_good(self):
        self.is_good = True
        for neuron in self.neurons:
            neuron.bias = neuron.good_bias
            for synapse in neuron.synapses:
                synapse.weight = synapse.good_weight
                synapse.decay = synapse.good_decay

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

    def mutation(self, running=False):
        self.is_good = False
        if running:
            mutation_rate = {'param': 0.3, 'struct': 0}
        else:
            mutation_rate = {'param': 0.5, 'struct': 0.5}
        rate = tanh(self.effect * 5)
        for neuron in self.neurons:
            if random.random() <= mutation_rate['param']:
                neuron.bias = safe_param(neuron.good_bias + (random.random() - 0.5) * rate)
            for synapse in neuron.synapses:
                if random.random() <= mutation_rate['param']:
                    synapse.weight = safe_param(synapse.good_weight + (random.random() - 0.5) * rate)
                if random.random() <= mutation_rate['param']:
                    synapse.decay = safe_param(synapse.good_decay + (random.random() - 0.5) * rate)
        if random.random() <= mutation_rate['struct'] and self.effect > 0:
            degree = len(self.neurons)
            i = random.randint(0, degree - 1)
            j = random.randint(0, self.input_dimension + degree - 1)
            exist = False
            for synapse in self.neurons[i].synapses:
                if synapse.sid == j:
                    exist = True
                    break
                elif synapse.sid > j:
                    break
            if not exist:
                if j < self.input_dimension:
                    neu = self.feel[j]
                else:
                    neu = self.neurons[j - self.input_dimension]
                self.neurons[i].synapses.append(Synapse(neu,
                                                        sid=j,
                                                        weight=0,
                                                        decay=random.random()))
                self.neurons[i].synapses.sort(key=lambda x: x.sid)


class Synapse:
    def __init__(self, neu, sid=0, weight=1.0, decay=0, learn=True):
        self.neu = neu
        self.sid = sid
        self.weight = weight
        self.decay = decay
        self.good_weight = weight
        self.good_decay = decay
        self.learn = learn
        self.live_time = 0
        self.active = False
        self.decayed = False
        self.value = 0

    def count(self):
        weighted_value = self.neu.value * self.weight
        if self.learn and self.live_time < MAX_TIME:
            self.live_time += 1
        self.decay = math.fabs(self.decay)
        if self.decay == 0 or self.decay == 1:
            self.value = weighted_value
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
                if math.fabs(weighted_value) >= 0.9:
                    if not self.decayed:
                        self.active = True
                        self.value = weighted_value
                else:
                    self.decayed = False
                    self.value = weighted_value
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
        self.good_bias = bias
        self.synapses = []
        self.learn = learn
        self.value = 0

    def count(self):
        s_sum = 0
        for s in self.synapses:
            s_sum += s.count()
        s_sum -= self.bias
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
        for synapse in self.synapses:
            if synapse.live_time >= MAX_TIME and abs(synapse.weight) <= 0.01:
                self.synapses.remove(synapse)
