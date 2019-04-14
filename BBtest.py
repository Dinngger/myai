#!usr/bin/python
import BugBrain as BB
import my_env


def test1():
    env = my_env.make("Remember")
    env.reset()
    brain = BB.Brain()
    brain.neurons.append(
        BB.Neuron('Step', sid=0, bias=0))
    brain.neurons.append(
        BB.Neuron('Step', sid=1, bias=0))
    brain.neurons[0].synapses.append(
        BB.Synapse(brain.neurons[1], weight=-1, decay=0.001))
    brain.neurons[1].synapses.append(
        BB.Synapse(brain.neurons[0], weight=-1, decay=0.001))
    for i in range(100):
        brain.work()
        env.step([0])
        env.render()
        brain.draw()
        print("t: {}  a: {}  b: {}".format(i, brain.neurons[0].value, brain.neurons[1].value))


if __name__ == '__main__':
    test1()
