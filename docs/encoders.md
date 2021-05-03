### Encoder details
This file describes the details of the encoders in the paper.


Contrastive learning is the important part to implement in this reproduction repo. 
The paper doesn't use pretraining but calculates the loss directly during the RL training.  
They create keys and queries from the latest stack of images and replay buffer (Could also be only the replay buffer).
Useful quotes from the paper about contrastive learning implementation:
>  The query observations oq are treated as the
anchor while the key observations ok contain the positive and
negatives, all constructed from the minibatch sampled for the RL
update.


> To specify a contrastive learning objective,
we need to define (i) **the discrimination objective** (ii) the
transformation for **generating query-key observations** (iii)
the embedding procedure for **transforming observations into
queries and keys** and (iv) the inner product used as a **similarity 
measure** between the query-key pairs in the contrastive
loss. The exact specification these aspects largely determine
the quality of the learned representations.


> We use a momentum encoding procedure for
targets similar to MoCo (He et al., 2019b) which we found
to be better performing for RL. Finally, for the InfoNCE
score function, we use a bi-linear inner product similar to
CPC (Contrastive Predictive Coding) (van den Oord et al., 2018)  
> ...  
> InfoNCE is an unsupervised loss
that learns encoders fq and fk mapping the raw anchors
(query) xq and targets (keys) xk into latents q = fq(xq) and
k = fk(xk), on which we apply the similarity dot products.

The encoders are setup using momentum contrast (MoCo): using exponentially moving average (momentum averaged) version of the query encoder 
for encoding the keys. To recap, the key encoder is only used for calculating the contrastive loss. The query encoder is also used in the RL pipeline.
The key encoder also doesn't calculate the gradient when encoding, since the parameters are updated using the exponentially moving average of the query encoder.


Discriminative objective:
`
CURL therefore uses instance discrimination rather than patch discrimination.
`  
Normally using Contrastive Predictive Coding groups of image patches are separated 
by a carefully chosen spacial offset, adding extra hyperparameters and also wall-clock training time which is why the papers chose a different path.

Negatives are coming from other pictures in the replay buffer:
> Similar to instance discrimination in the image setting (He
et al., 2019b; Chen et al., 2020), the anchor and positive
observations are two different augmentations of the same
image while negatives come from other images. CURL primarily relies
on the random crop data augmentation, where a
random square patch is cropped from the original rendering.
> A simple instantiation of contrastive learning is Instance
Discrimination (Wu et al., 2018) wherein a query and key
are positive pairs if they are data-augmentations of the same
instance (example, image) and negative otherwise.

`The aspect ratio for cropping is 0.84, i.e., they crop a 84 x 84 image from a 100 x 100 image. They applay the same crop coordinates for the stack of frames.`

The architecture given in pseudocode by the paper:  
4 Conv layers with 32 feature maps, kernel size of 3, first layer a stride of 2 the remaining stride of 1 and all with Relu activation. 
These layers followed by a hidden linear layer of 1024 units and output layer of 50 units ending with LayerNorm and tanh activation.
```
def encode(x,z_dim):
"""
ConvNet encoder
args:
B-batch_size, C-channels
H,W-spatial_dims
x : shape : [B, C, H, W]
C = 3 * num_frames; 3 - R/G/B
z_dim: latent dimension
"""
x = x / 255.
# c: channels, f: filters
# k: kernel, s: stride
z = Conv2d(c=x.shape[1], f=32, k=3, s=2)])(
x)
z = ReLU(z)
for _ in range(num_layers - 1):
z = Conv2d((c=32, f=32, k=3, s=1))(z)
z = ReLU(z)
z = flatten(z)
# in: input dim, out: output_dim, h:
hiddens
z = mlp(in=z.size(),out=z_dim,h=1024)
z = LayerNorm(z)
z = tanh(z)
```