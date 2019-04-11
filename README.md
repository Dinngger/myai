# myai
This program uses the BugBrain to build AIs to play in the OpenAi gym.
## Introduction
### BugBrain
- This neuron network comes from the game 'BugBrain'. Import BugBrain to use it.
```python
    import BugBrain as BB
```
- Use class Brain to build a brain
```python
    brain = BB.Brain()
```
- Then add neurons to the barin. The neuron has three different activation functions. They are 'Step', 'Linear', and 'Sigmoid'.
```python
    brain.neurons.append(BB.Neuron('Step'))
```
- Create a inputnode and a synapse to the neuron.
```python
    inputnode = BB.InputNode()
    brain.neurons[0].synapses.append(BB.Synapse(inputnode))
```
- In this way there will be a link from the inputnode to the neuron. Then you can change the weight of the synapse or the bias of the neuron. Set the inputnode and run the barin.
```python
    inputnode.value = 1
    brain.work()
    output = brain.neurons[0].value
```
### worm
- Now the worm is used to play the 'CartPole-v1'. There is only one neuron in its brain. I use evolutionary algorithms to train the worms.
- I am trying to enable them to evolute their brain's shape automatically, and then they can play more difficult games.
- The BugBrain has a huge potential. Next time I will try to enable the decay function. Then the worms can remember things.
## How to Use
- install gym
- run 'worm.py'
- you can see after many generations the worm can play well.
## Environments need
gym