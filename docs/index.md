# Reinforcement Learning on your home computer with CURL
##### Gijs Koning (4453484) g.koning@student.tudelft.nl
##### Chiel de Vries (4571983) c.devries-1@student.tudelft.nl

## Introduction

Deep learning is an amazing new tool in the world of computer science. New techniques are developed daily and there is currently no end in sight. One problem encountered by many students in this field is the insane amount of computing power that is necessary for some models. For student that want to study the field this can be a big hurdle. 

CURL (Contrastive Unsupervised representations for Reinforcement Learning) [CITE] is a model that performs representation learning for reinforcement learning (RL) agents. It can learn to do complex tasks from raw pixel data.  It promised to be more sample efficient than currently available models. This work aims to recreate the performance of the already efficient CURL model with lower compute settings, like reduced batch size and replay buffer size. This frees up a lot of memory which allows it to be run on machines that would be available for a student. 

This work replicates CURL and  compares with another unsupervised pixel-based RL model: PixelSAC. They are run on the same settings to level the playing field and provide a fair comparison. 

bit about results

## Model
This section will briefly introduce The implementation of CURL and it's components. CURL uses a contrastive representation learner that provides meaningful representations from raw pixel data. The representations are then passed to a RL model, which is Soft Actor Critic for this work.

![CURL-schematic](images/CURL.png) 

### Contrastive Learning
Contrastive learning is a form of unsupervised representation learning. It learns to distinguish between different augmentations of the same images and augmentations of other images. A schematic overview of the system can be viewed above. The way it works is as follows:

First, the observation is augmented twice, once as a query and once as a key. In this case augmentation means taking a random crop of 84 by 84 pixels (the original size is 100 by 100 pixels). Most contrastive learning algorithms use just one image as an input, but since CURL is designed for reinforcement learning it benefits from having temporal information as well. Therefore, a stack of three images is used which are simply stacked together in the same dimension as the RGB channels. Thus, creating a stack of size 9x100x100. The whole stack is cropped in the same way during augmentation. 

Second, the query and the key are encoded to latent vectors of size 50. They are encoded by two different encoders, The key-encoder being a momentum updated version of the query-encoder. This means the weights of the key-encoder are updated by an exponential moving average (EMA) of the query encoder conform the formula below. The encoder itself is a simple set of 4 convolutional layers with ReLUs followed by an MLP with one hidden layer of size 1024, with layernorm and tanh non-linearity. 

![momentum](images/momentum.PNG)

Third, a similarity is calculated between the query and a set of keys. The goal is to ensure that the query is most similar to it's corresponding key, called the _positive_, and to minimise the similarity with the other keys, called the _negatives_. The negatives are the keys of the other images in the current batch. The similarity measure used is bilinear similarity. The loss function used to train the system is the InfoNCE loss[CITE]. It can be interpreted as the log loss of a K-way softmax classifier where the label is the positive (k+), and W is a learnable parameter.

![InfoNCE](images/InfoNCE.PNG)

This is the basic setup of the contrastive learning CURL uses. However, it has one more ace up its sleeve. The encoder is not only updated using the contrastive loss, but also using the loss created in the connected RL agent. This enables the representations created by the encoder to be useful for the specific task the agent aims to teach itself. 

CURL is designed to be able to work with any reinforcement learning algorithm. Srinivas et al. [CITE] use two algorithms in their work: SAC and Rainbow DQN. This work uses just the SAC algorithm since the performance of CURL is tested on a DMC task which needs continuous input values. The way SAC works is discussed in the next section.

### Soft Actor Critic
SAC [CITE] is a model-free deep reinforcement learning algorithm. The aim of this algorithm to improve the sample efficiency and stability compared with other state of the art RL frameworks. The full explanation of the workings of SAC is outside of the scope of this work, but it does aim to provide some intuition on how it works for the uninitiated. 

RL algorithms are agents. They perceive an environment and use those observations to decide what action to perform on the environment. They decide what action to use based on a reward they receive from the environment upon performing the action. The agent aims to maximize this reward over time. With Deep RL algorithms the internals of the agents are based on deep neural networks. 

SAC builds on previous actor critic methods and has two main components: The _actor_ and the _critic_. The actor learns to sample actions from the _policy_ and the critic learns to improve the policy based on the reward of the action and the current state of the environment. The policy is a function that determines the strategy of the agent. Together the actor and the critic aim to maximize the effectiveness of the policy to generate rewards by using a method called maximum entropy reinforcement learning. In CURL the gradient of the critic network is used to update the encoder during training. 

To reiterate, the description above is greatly oversimplifying the methods used in SAC. An in depth description can be found in the original paper by Haarnoja et al. [CITE]

## Experimental Setup

### Deep Mind Control

### Training

## Results

## Conclusion and Discussion

## References
