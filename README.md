## Reproducing CURL Project plan
We want to reproduce the unsupervised learning steps of the paper and reuse an existing SAC Reinforcement learning algorithm.  
https://arxiv.org/pdf/2004.04136.pdf
Why do we want to do this? We want to learn more about Reinforcement learning and unsupervised learning. This paper combines both. ...

Where is the data? The data is already available and coming from simulating environment steps in the DMControl or Atari environment. _needs references_  
How much time does training take? We don't know yet for sure but is expected to be reasonable.  

### Plan
- Week 2 (start): Create project plan, read paper in detail and startup repo.
- Week 3: Start implementation unsupervised model with RL algorithm (Need to find SAC algorithm) and get first images from DMcontrol suite. 
- Week 4: Work further on implementation possibly get first results.
- Week 5: Work further on implementation. Reflect on current progress and replan coming weeks. See if any research questions come up.
- Week 6: Training different tasks. Find possible improvements.
- Week 7: Training etc. Start basics for blog.
- Week 8: Continue...
- Week 9: Continue...
- Week 10: 90% of blog done and final runs.
- Week 11: Blog finished and presentation created.

### Implementation decisions
- Preferably we want to use the DMControl suite since it has a bit more complex and fun environments since it has only continues control environments.
- What kind of environments do we want to use? Preferably the ones that also worked good in the paper. Answer: ...
- Using Pytorch as Deep Learning framework.

### What we want to learn
- Unsupervised learning.
- The use of Reinforcement Learning.
- How to document a Deep Learning project process.

### Currently working on
- Reading the paper in detail
- Setting up basics for the project: Finding the RL algorithm, setting up model class and setup training loop.

### Meetings
3 May next meeting. Things to discuss:
- CURL is an improvement on Pixel SAC. Would it be good to also implement the baseline? It is mostly swapping the first couple of CNN layers.
- ...


## Other related papers and information
- Reinforcement Learning with Augmented Data
- Learning Invariant Representations for Reinforcement Learning without Reconstruction
- Decoupling Representation Learning from Reinforcement Learning
- data-efficient reinforcement learning with self-predictive representations
- Soft Actor Critic (SAC) (Haarnoja et al., 2018) https://arxiv.org/abs/1801.01290  
  Documentation: https://spinningup.openai.com/en/latest/algorithms/sac.html

