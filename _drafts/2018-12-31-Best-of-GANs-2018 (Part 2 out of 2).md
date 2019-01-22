---
layout: post
title: "The best of GAN papers in the year 2018 part 2"
author: "Damian Bogunowicz"
categories: blog
tags: [computer vision, neural networks, generative adversarial networks]
image: gan2.jpg
---
The cover image by courtesy of [Juli Odomo](https://www.odomojuli.com).

As a follow up to my previous post, where I discussed three major contributions to GANs (Generative Adversarial Networks) 
domain, I am happy to present another three interesting research papers from 2018. Once again, the order is purely random and the choice
very subjective.

1. __Large Scale GAN Training for High Fidelity Natural Image Synthesis__ - DeepMind's BigGAN uses the power of hundreds of cores of a Google TPU v3 Pod to create high-resolution images on a large scale.
2. __The relativistic discriminator: a key element missing from standard GAN__ - the author proposes to improve the fundamentals of GANs by introducing an improved discriminator.
3. __empty__ - 



## [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/pdf/1809.11096.pdf)

### Details
The paper has been submitted on 28.09.2018. You can easily [run BigGAN](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/biggan_generation_with_tf_hub.ipynb) using Google Coolab!.

### Main idea:

Even though the progress in the domain of GANs is impressive, image generation using Deep Neural Networks remains difficult. Despite the great interest in this field, I believe that there is a lot of untapped potential when it comes to generating images. One of the ways to track the progress of GANs and measure their quality is [Inception Score](https://arxiv.org/abs/1606.03498) (IS). This metric considers both quality of generated images as well as their diversity. Using the example of 128 by 128 images from [ImageNet dataset](http://www.image-net.org/) as our baseline, the real images from the dataset achieve $$IS = 233$$. While the state-of-the-art was estimated at $$IS = 52.5$$, BigGAN has set the bar at $$IS = 166.3$$! How is this possible?
The authors show how GANs can benefit from training at large scale. Leveraging the immense computational resources allows for dramatic boost of performance, while keeping the training process relatively stable. This allows for creation of high resolution images (512x512) of unparalleled quality. Among many clever solutions to instability problem, this paper also introduces the truncation trick, which I have already discussed in part 1 of my summary (publication __A Style-Based Generator Architecture for Generative Adversarial Networks__).

### The method:

In contrast to other papers I evaluated, the significance of this research does not come from any significant modification to the GAN framework. Here, the major contribution comes from using massive amounts of computational power available (courtesy of Google) to make the training more powerful. This involves using larger models (4-fold increase of parameter number with respect to prior art) and larger batches (increase by almost order of magnitude). This turns out to be very beneficial:
1. Using large batch sizes (2048 images in one batch) allows every batch to cover more modes. This way the discriminator and generator benefit from better gradients.
2. Doubling the width (number of channels) in every layer increases the capacity of the model and thus contributes to much better performance. Interestingly, increasing the depth has negative influence on the performance.
3. Additional use of class embeddings accelerates the training procedure. Class embeddings mean conditioning the output of the generator on dataset's class labels.
4. Finally, the method also benefits from hierarchical latent spaces - injecting the noise vector $$\textbf{z}$$ into multiple layers rather then solely at the initial layer. This not only improves performance of the network, but also accelerates the training process.

### Results:

Large scale training allows for superior quality of generated images. However, it comes with its own challenges, such as instability. The authors show, that even though the stability can be enforced through regularization methods (especially on the discriminator), the quality of the network is bound to suffer. The clever workaround is to relax the constraints on the weights and allow for training collapse at the later stages. Here, we may apply the early stopping technique to pick the set of weights just before the collapse. Those weights are usually sufficiently good to achieve impressive results.

{:refdef: style="text-align: center;"}
![alt text](/assets/5/1.png)
{: refdef}
<em> One generated image and its nearest neighbours from ImageNet dataset. Which image is artificially generated? The burger in the top left corner...</em> 

{:refdef: style="text-align: center;"}
![alt text](/assets/5/2.png)
{: refdef}
<em> Great interpolation ability in both class and latent space confirms that the model does not simply memorize data. It is capable of coming up with its own, incredible inventions!</em> 

{:refdef: style="text-align: center;"}
![alt text](/assets/5/3.png)
{: refdef}
<em> While it may be tempting to cherry-pick the best results, the authors of the paper also comment on the failure cases. While easy classes such as a) allow for seamless image generation, difficult classes b) are tough for the generator to reproduce. There are many factors which influence this phenomenon e.g. how well the class is represented in the dataset or how sensitive our eyes to particular objects. While small flaws in the landscape image are unlikely to draw our attention, we are very vigilant towards "weird" human faces or poses. </em>

## [The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/pdf/1807.00734.pdf)

### Details
The paper has been submitted on 02.06.2018. The reason why I am impressed by this work is because it seems that the whole job was done by one person. The author thought about everything - writing a short blog post about [her invention] (https://ajolicoeur.wordpress.com/relativisticgan/), publish well documented [source code](https://github.com/AlexiaJM/RelativisticGAN) and spark an interesting [discussion on reddit](https://www.reddit.com/r/MachineLearning/comments/8vr9am/r_the_relativistic_discriminator_a_key_element/).

### Main idea:

In standard generative adversarial networks, the discriminator $$D$$ estimates the probability of the input data being real or not. The generator $$G$$ tries to increase the probability that generated data is real. During training, in every iteration, we input two equal-sized batches of data into the discriminator: one batch comes from a real distribution $$\mathbb{P}$$, the other from fake distribution $$\mathbb{Q}$$. 
This valuable piece of information, that half of the examined data comes from fake distribution is usually not conveyed in the algorithm. Additionally, in standard GAN framework, the generator attempts to make fake images look more real, but there is no notion that the generated images can be actually “more real” then real images. The author claims that those are the missing pieces, which should have been incorporated into standard GAN framework in the first place. Due to those limitations, it is suggested that training the generator should not only increase the probability that fake data is real but also decrease the probability that real data is real. This observation is also motivated by the IPM-based GANs, which actually benefit from the presence of relativistic discriminator.


### The method:

In order to shift from standard GAN into “relativistic” GAN, we need to modify the discriminator. A very simple example of a Relativistic GAN (RGAN) can be conceptualized in a following way:

In __standard formulation__, the discriminator is a function $$D(x) = \sigma(C(x))$$. $$x$$ is an image (real or fake), $$C(x)$$ is a function which assigns a score to the input image (evaluates how realistic $$x$$ is) and $$\sigma$$ translates the score into a probability between zero to one. If discriminator receives an image which looks fake, it would assign a very low score and thus low probability e.g. $$D(x) = \sigma(-10)=0$$. On the contrary, real-looking input gives us high score and high probability e.g. $$D(x) = \sigma(5)=1$$.

Now, in __relativistic GAN__, the discriminator estimates the probability that the given real data $$x_r$$ is more realistic then a randomly sampled fake data $$x_f$$:

$$D(\widetilde{x}) = \sigma(C(x_r)-C(x_f))$$

To make the relativistic discriminator act more globally and avoid randomness when sampling pairs, the author builds up on this concept to create a Relativistic average Discriminator (RaD). 

$$\bar{D}(x)=\begin{cases}
sigma(C(x)-\mathop{\mathbb{E}}_{x_{f}\sim\mathbb{Q}}C(x_{f})), & \text{if $x_f$ is real}\\
sigma(C(x)-\mathop{\mathbb{E}}_{x_{f}\sim\mathbb{P}}C(x_{f})), & \text{if $x_r$ is fake}.
 \end{cases}$$

This means that whenever the discriminator $$D\hat$$ receives a real image, it evaluates how is this image more realistic that the average fake image from the batch in this iteration. Analogously, $$D\hat$$ receives a image, it is being compared to an average of all real images in a batch. This formulation of relativistic discriminator allows us to indirectly compare all possible combinations of real and fake data in the minibatch, without introducing quadratic time complexity of the algorithm. 


### Results:

{:refdef: style="text-align: center;"}
![alt text](/assets/5/4.png)
{: refdef}
<em>The diagram shows an example of the discriminator’s output in standard GAN: $$P(x_r \text{is real|) = \sigma(C(x_r)))$$ 
and RaD: P(x_r \text{is real}|C(x_f)) = \sigma(C(x_r) − C(x_f))). $$x_f$$ are dogs images while $$x_r$$ are pictures of bread.
I think that this example gives a very good intuitive understanding of implications coming along with relativistic disciminator.
{:refdef: style="text-align: center;"}
![alt text](/assets/5/5.png)
{: refdef}
<em>Artificially created cats (128x128 resolution), the output from RaLSGAN. Not only the standard LSGAN produces less realistic images, it  is also much more unstable.
 
I have the impression that this paper is only an introduction to the idea of relativistic discriminator. The experiments indicate, that the approach may help with many problems such as stability, difficulty in creating plausible looking outputs. It may also accelerate the training speed. I really love the fact, that the author has questioned a very fundamental element of the puzzle. It is exciting to see, that even though this paper came out in June, there are already state-of-the-art publications which take advantage of relativistic discriminators. 
 
 
## [Evolutionary Generative Adversarial Networks](https://arxiv.org/abs/1803.00657)

### Details
The paper has been submitted on 1.03.2018.

### Main idea:
In the classical setting GANs are being trained by alternately updating a generator and discriminator using back-propagation. This two-player minmax game is being implemented by utilizing the cross-entropy mechanism in the objective function. 
The authors of E-GAN propose the alternative GAN framework which is based on evolutionary algorithms. They restate loss function in form of an evolutionary problem. The task of the generator is to undergo constant mutation under the influence of the discriminator. According to the principle of "survival of the fittest", one hopes that the last generation of generators would "evolve" in such a way, that it learns the correct distribution of training samples.

### The method:


<em>All the figures are taken from the publications I refer to in my blog post<em>
 




