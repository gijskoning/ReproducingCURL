# Reinforcement Learning on your home computer with CURL
##### Gijs Koning (4453484) g.koning@student.tudelft.nl
##### Chiel de Vries (4571983) c.devries-1@student.tudelft.nl

## Introduction

Deep learning is an amazing new tool in the world of computer science. New techniques are developed daily and there is currently no end in sight. One problem encountered by many students in this field is the insane amount of computing power that is necessary for some models. For student that want to study the field this can be a big hurdle. 

CURL (Contrastive Unsupervised representations for Reinforcement Learning) [CITE] is a  model that performs representation learning for reinforcement learning agents. It can learn to do complex tasks from raw pixel data.  It promised to be more sample efficient than currently available models. This work aims to recreate the performance of the already efficient CURL model with lower compute settings, like reduced batch size and replay buffer size. This frees up a lot of memory which allows it to be run on machines that would be available for a student. 

This work replicates CURL and  compares with another unsupervised pixel-based reinforcement learning model: PixelSAC. They are run on the same settings to level the playing field and provide a fair comparison.

bit about results

## Model
This section will briefly introduce The implementation of CURL and it's components. CURL uses a contrastive representation learner that provides meaningful representations from raw pixel data. The representations are then passed to a reinforcement learning model, which is Soft Actor Critic for this work.

![CURL-schematic](images/CURL.png) 
### Contrastive Learning
Contrastive learning is a form of unsupervised representation learning. It learns to distinguish between different augmentations of the same images and augmentations of other images. A schematic overview of the system can be viewed above. The way it works is as follows. 

First, the observation is augmented twice, once as a query and once as a key. In this case augmentation means taking a random crop of 84 by 84 pixels (the original size is 100 by 100 pixels). 

Second, the query and the key are encoded to latent vectors of size 50. They are encoded by two different encoders, The key-encoder being a momentum updated version of the query-encoder. 

Third, a similarity is calculated between the query and a set of keys. The goal is to ensure that the query is most similar to it's corresponding key, called the _positive_, and to minimise the similarity with the other keys, called the _negatives_. The negatives are the keys of the other images int the current batch. the similarity measure used is bilinear similarity.

The loss function used to train the system is the InfoNCE loss[CITE]. It can be interpreted as the log loss of a K-way softmax classifier where the label is the positive. 
![InfoNCE](images/InfoNCE.png)






### SAC

## Experimental Setup



### Deep Mind Control

### Training

## Results

## Conclusion

## References
