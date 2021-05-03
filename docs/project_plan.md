# Project Plan

This page contains the initial project plan we made in week 2. 

## Goal of the Reproduction
The main goal is to implement the [CURL](https://arxiv.org/pdf/2004.04136.pdf) architecture by ourselves and test it, in combination with an existing [SAC](https://arxiv.org/pdf/1801.01290.pdf) implementation. 
This will mostly be implementing the encoder model and the exponentially moving average encoder by only using details of the paper. We will reuse a part of their repo using the util functions and the RL algorithm since this is not computer vision related.

The paper has performed many experiments on lots of different environments. It uses 16 environments from DMControl and all 26 from Atari. Since we only have limited time and resources, we plan to only test a couple (2-3) environments from DMControl. Which environments we will used specifically will be determined at a later stage. If time permits it we would love to test on more environments, but we think this is good as a start. We choose DMControl over Atari because we think the DMControl environments are more interesting since they have a continuous action space instead of a (mostly) discrete action space for the Atari games.

## Implementation Details
- We use Pytorch as our Deep Learning framework.
- We use the DeepMind Control Suite to test our network. [link](https://arxiv.org/pdf/1801.00690.pdf)
- SAC implementation: TBA 

## Expected Memory usage and Training Time 
The authors of the paper use a batch size of 512 for training. The forward pass of the encoder should then use about 300 MB of memory for its four convolutional layers, two linear layers and activations. We're not quite sure how the memory usage works for the momentum encoder that encodes the keys, but in the worst case this is still only double the amount of memory, which is perfectly reasonable. The SAC network is fairly small and only has 2 hidden layers of size 256. Thus, this takes about 0.5 MB of memory.

Time cost for training is expected to be reasonable, but we do not know that for sure. That's we we will determine at a later stage how many environments we will test.

## Further Research Options
If our work turn out to be too trivial, we purpose we extend our initial plan by one or more of the following options:
- We  test more environments. Either from the DMControl suite or Atari. 
- We Make our own implementation of SAC on top of CURL and compare this with the  SAC implementation used by the authors.
- Do the same additional experiments as the paper: detaching the encoder training from the SAC algorithm learning and then visualizing what happens with the learned kernels. 
- We find a paper that extend CURL in some way and reproduce that as well.
  - Or extend using "Decoupling Representation Learning from Reinforcement Learning"
  - Could extend the implementation by the paper "Improving Computational Efficiency in Visual
    Reinforcement Learning via Stored Embeddings".

## Planning
- __Week 2 (start):__ Create project plan, read paper in detail and startup repo.
- __Week 3:__ Start implementation unsupervised model with RL algorithm (Need to find SAC algorithm) and get first images from DMcontrol suite. 
- __Week 4:__ Work further on implementation possibly get first results.
- __Week 5:__ Work further on implementation. Reflect on current progress and replan coming weeks. See if any research questions come up.
- __Week 6:__ Training different tasks. Find possible improvements.
- __Week 7:__ Training etc. Start basics for blog.
- __Week 8:__ Continue...
- __Week 9:__ Continue...
- __Week 10:__ 90% of blog done and final runs.
- __Week 11:__ Blog finished and presentation created.