# Log 
This page hosts the log of both our team members. 

## Gijs

### Week 2 (26-04-2021)
- Mon: Meeting with TA
- Saturday: Add basic file structure and installation instructions.  

Contrastive learning is the important part to implement in this reproduction repo. 
The paper doesn't use pretraining but calculates the loss directly during the RL training.  
They create keys and queries from the latest stack of images and replay buffer (Could also be only the replay buffer).
Useful quotes from the paper about contrastive learning implementation:
```
 The query observations oq are treated as the
anchor while the key observations ok contain the positive and
negatives, all constructed from the minibatch sampled for the RL
update.
```
```
To specify a contrastive learning objective,
we need to define (i) the discrimination objective (ii) the
transformation for generating query-key observations (iii)
the embedding procedure for transforming observations into
queries and keys and (iv) the inner product used as a similarity 
measure between the query-key pairs in the contrastive
loss. The exact specification these aspects largely determine
the quality of the learned representations.
```
```
We use a momentum encoding procedure for
targets similar to MoCo (He et al., 2019b) which we found
to be better performing for RL. Finally, for the InfoNCE
score function, we use a bi-linear inner product similar to
CPC (Contrastive
Predictive Coding) (van den Oord et al., 2018)...
```
Discriminative objective:
`
CURL therefore uses instance discrimination rather than patch discrimination.
`. Groups of image patches are separated by a carefully chosen spacial offset, adding extra hyperparameters and also wall-clock training time.

Negatives are coming from other pictures in the replay buffer:
```
Similar to instance discrimination in the image setting (He
et al., 2019b; Chen et al., 2020), the anchor and positive
observations are two different augmentations of the same
image while negatives come from other images. CURL primarily relies
on the random crop data augmentation, where a
random square patch is cropped from the original rendering.
```
```
A simple instantiation of contrastive learning is Instance
Discrimination (Wu et al., 2018) wherein a query and key
are positive pairs if they are data-augmentations of the same
instance (example, image) and negative otherwise.
```
`The aspect ratio for cropping is 0.84, i.e., they crop a 84 x 84 image from a 100 x 100 image. They applay the same crop coordinates for the stack of frames.`

### Week 3 (03-05-2021)

### Week 4 (10-05-2021)

### Week 5 (17-05-2021)

### Week 6 (24-05-2021)

### Week 7 (31-05-2021)

### Week 8 (07-06-2021)

### Week 9 (14-06-2021)

### Week 10 (21-06-2021)

### Week 11 (28-06-2021)

## Chiel

### Week 2 (26-04-2021)
- Mon: Meeting with TA
- Fri: Read paper, write motivation, clean documentation, estimate memory usage.

### Week 3 (03-05-2021)

### Week 4 (10-05-2021)

### Week 5 (17-05-2021)

### Week 6 (24-05-2021)

### Week 7 (31-05-2021)

### Week 8 (07-06-2021)

### Week 9 (14-06-2021)

### Week 10 (21-06-2021)

### Week 11 (28-06-2021)

