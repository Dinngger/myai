#!usr/bin/python
import matplotlib.pyplot as plt
import networkx as nx
import math
# bias and weight should be in [-1, 1]
# decay should be in [0, 1]
# the larger decay is, the faster the decay are.
G = nx.DiGraph()
plt.ion()
types = {'Step': 0, 'Linear': 1, 'Sigmoid': 2, 'Tanh': 3}


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def tanh(x):
    expx = math.exp(x)
    exp_x = math.exp(-x)
    return (expx - exp_x) / (expx + exp_x)


def learn_rate(x):
    x = abs(x)
    if x > 1:
        x = 1
    x = (x - 0.5) / 4
    # input 0 will return 0.998, after 1000 times the weight will be about 0.1
    return pow(tanh(x), 3) + 1


def safe_param(x):
    return 1 if x > 1 else (-1 if x < -1 else x)


class Brain:
    def __init__(self):
        self.neurons = []

    def work(self):
        for neuron in self.neurons:
            neuron.count()

    def draw(self, inputNodes, reward):
        G.clear()
        plt.clf()
        node_colors = []
        edge_colors = []
        for neuron in self.neurons:
            G.add_node(neuron.sid)
            node_colors.append((neuron.value + 1) / 2)
        for in_node in inputNodes:
            G.add_node(in_node.sid)
            node_colors.append((in_node.value + 1) / 2)
        G.add_node(reward.sid)
        node_colors.append((reward.value + 1) / 2)
        for neuron in self.neurons:
            for synapse in neuron.synapses:
                G.add_edge(synapse.neu.sid, neuron.sid)
                edge_colors.append((synapse.value + 1) / 2)
        nx.draw_planar(G, with_labels=True,
                       node_color=node_colors,
                       edge_color=edge_colors,
                       cmap=plt.get_cmap('cool'),
                       edge_cmap=plt.get_cmap('cool'))
        plt.pause(0.001)

    def parameter(self):
        param = []
        for neuron in self.neurons:
            param.append(neuron.bias)
            for synapse in neuron.synapses:
                param.append(synapse.weight)
                param.append(synapse.decay)
        return param

    def updateParam(self, param):
        i = 0
        for j in range(len(self.neurons)):
            self.neurons[j].bias = safe_param(param[i])
            i += 1
            for k in range(len(self.neurons[j].synapses)):
                self.neurons[j].synapses[k].weight = safe_param(param[i])
                i += 1
                self.neurons[j].synapses[k].decay = safe_param(param[i])
                i += 1


class Synapse:
    def __init__(self, neu, sid=0, weight=1.0, decay=0):
        self.neu = neu
        self.sid = sid
        self.weight = weight
        self.decay = decay
        self.learn = False
        self.active = False
        self.decayed = False
        self.value = 0

    def count(self):
        val_w = self.neu.value * self.weight
        self.weight = safe_param(self.weight * learn_rate(val_w))
        if self.decay == 0:
            self.value = val_w
        else:
            if self.active:
                self.value = self.value * self.decay
                if math.fabs(self.value) > math.fabs(val_w):
                    self.value = val_w
                if math.fabs(self.value) <= 0.01:
                    self.active = False
                    self.decayed = True
                    self.value = 0
            else:
                if math.fabs(val_w) >= 0.9:
                    if not self.decayed:
                        self.active = True
                        self.value = val_w
                else:
                    self.decayed = False
                    self.value = val_w
        return self.value


class InputNode:
    def __init__(self, sid=0):
        self.sid = sid
        self.value = 0


class Neuron:
    def __init__(self, the_type, sid=0, bias=0.5):
        self.__type = types[the_type]
        self.sid = sid
        self.bias = bias
        self.synapses = []
        self.learn = False
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
        else:
            self.value = tanh(s_sum)
        for synapse in self.synapses:
            if synapse.weight == 0:
                self.synapses.remove(synapse)
