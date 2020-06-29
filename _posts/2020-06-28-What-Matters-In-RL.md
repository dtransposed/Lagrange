---
layout: post
crosspost_to_medium: true
title: "Research Paper Summary - What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study"
author: "Damian Bogunowicz"
categories: blog
tags: [reinforcement learning, machine learning, summary, research]
image: whatmatters.png

---


Recent paper from Google Brain team, [What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990), tackles one of the notoriously neglected problems in deep Reinforcement Learning (deep RL). I believe this is a pain point both for RL researchers and engineers: 

> Out of dozens of RL algorithm hyperparameters, which choices are actually important for the performance of the agent?
>
> In other words, the goal is to understand how the different choices affect the final performance of an agent and to derive recommendations for these choices.

In this blog post I will go through the key takeaways from the paper. My goal is to present the learnings and briefly comment on them. **I believe that many of those findings may be hugely beneficial for any machine learning engineer who dips his toes in the field of RL.** 

Kudos for the Google Brain Team for compiling sort of a "check list" in their research paper. In every section they present the most important hyperparameters and suggest how one should go about initialising and tuning them. Even though this study is focused on the on-policy RL, the majority of learnings can be reused for off-policy algorithms. 

The reader can choose to quickly scam through the write-up to extract those "rules of thumb" but I recommend to read the source publication thoroughly. The study is really well documented with all the details explained.

# Introduction 

I believe that the publication may resonate with many engineers out there. Once it has been decided that our particular problem can be solved using deep RL, the first step is usually to set up the RL scaffolding. Here, one could either reimplement an algorithm from the paper, along with the necessary infrastructure or just use one of the available RL frameworks: be it [Catalyst](https://github.com/catalyst-team/catalyst), [TF-Agents](https://github.com/tensorflow/agents) , [Acme](https://deepmind.com/research/publications/Acme)  or [Ray](https://github.com/ray-project/ray). The second step is to design your environment. It will change over time and enrich our agent with the experience. Finally, the third step: run python script, monitor the training and tap yourself on the back. Tensorboard displays increasing average reward over time and our agent, slowly but surely, appears to acquire some form of intelligence...



...well, in reality, **you would face TONS of problems on the way.** Those could be:

1. The sheer amount of hyperparameters available for tuning. This includes both "ML-specific" choices (batch size, networks' architecture, learning rate), but also "RL-specific" (time discount $$\gamma$$, hyperparameters of advantage estimator). If your agent does not converge with default hyperparameters - bad luck. You will probably need to not only find out which hyperparameter is the key to your agent's performance, but also figure out its optimal value.
2. It is quite likely that there are strong interactions existing between certain hyperparameters (e.g batch size and learning rate are very much intertwined). This makes the tuning complexity significantly harder.
3. Same algorithms may be implemented differently in different research papers or RL frameworks. Your initial prototype of the PPO agent may be training fine. But later you would like to distribute your system using Ray framework, and their implementation of PPO could be slightly different. As the result, your previous hyperparameter configuration does not work for the current implementation.

Point 3. Brings me to the the paper [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/abs/2005.12729). This publication highlights the following problem: if your algorithm turns out to be superior to, let's say, current state-of-the-art, is it better because the algorithm is more clever, or you just used more favourable sets of hyperparameters (or simply just better code quality)?

**All those aforementioned issues could be (partially) avoided if we had a sound understanding of the importance of various hyperparameters**. If we could draw some general conclusions about the significance of a particular configurations perhaps we could be more aware of its consequences on our RL system. Additionally, we could be less prone to mistaking more favourable hyperparameter configuration for superior algorithm design.

# Experiments

For benchmarking the researchers employ five, diverse OpenAI Gym environments: Hopper-v1, Walker2d-v1, HalfCheetah-v1, Ant-v1 and Humanoid-v1. 

They run about 250 000 experiments to investigate design choices from eight thematic groups:

- *Policy Losses*
- *Networks Architecture*
- *Normalisation and Clipping*
- *Advantage Estimation*
- *Training Setup*
- *Timestep Handling*
- *Optimizers*
- *Regularization*

## Policy Losses

**Design decisions:** choice of different policy losses (e.g. PPO, policy gradient, V-MPO etc.) on the performance of the benchmark environments.

**Recommendation:**

1. In the on-policy setting, PPO policy loss is a recommendable default choice for majority of environments. 

2. The PPO clipping threshold should be set by default to $$0.25$$ and then tuned.

## Networks Architecture

**Design decisions:** structure/size of actor/critic, activation functions, initialisation of network weights, standard deviations of actions, transformations of sampled actions.

**Takeaways:**

1. Initialize the last policy layer with $$100\times$$ smaller weights. 

2. Use<em> softplus</em> to transform network input into action standard deviation and add a (negative) offset to its input to decrease the initial standard deviations of actions. Tune this offset if possible. 

3. Use <em>tanh</em> both as the activation function (if the networks are not too deep) and to transform the samples from the normal distribution to the bounded action space. 

4. Use a wide value MLP (no layers shared with the policy) but tune the policy width (it might need to be narrower then the value MLP).

**Comments:** 

1. Prior to reading this study, I had the impression that sharing some layers between actor and critic is beneficial for the agent. I reasoned that parameter sharing could accelerate training (gradient flowing from both the actor and the critic). However, I have learned that the norm of gradients flowing back from the actor and the critic could be at completely different scale. As a result, actor's gradients may destabilise critic's parameters and vice-versa. This is probably the reason for keeping policy and value network separate.
2. Surprisingly, it is better to use<em> tanh</em> activations rather than <em>ReLU</em> between actor's/critic's MLP layers. But only networks with limited capacity shall benefit from it. Anecdotically,<em> tanh</em> activation tends to be more reliable when you have smaller networks. It learns faster and is less sensitive to big differences in input features than <em>ReLU</em>. 
3. The authors repetitively stress the importance of weights initialisation in actor's final layer. Those weights should be very small, so that the initial action distribution is centred around zero with a small standard deviation. The authors claim that this boosts performance of an agent, but do not give an intuition why. Perhaps this way we avoid biasing the agent towards some certain, suboptimal trajectory ahead of time (from which the agent may not fully recover). 

## Normalisation and Clipping

**Design decisions:** observation normalisation, value function normalisation, per-minibatch advantage normalisation, gradient and observation clipping.

**Takeaways:**

1. Always use observation normalisation and check if value function normalisation improves performance. 

2. Gradient clipping might slightly help but is of secondary importance.

**Comments:** Not surprisingly, observation normalisation is crucial. This is true for most of the (un-) supervised machine learning algorithms. However, I am personally (anecdotally) quite sure that gradient clipping tends to help a lot in certain situations. But naturally, this could be more or less true depending on the particular environment.

## Advantage Estimation

**Design decisions:** choice of different advantage estimators (e.g. GAE, N-step) and their hyperparameters on the performance of the benchmark environments.

**Recommendation:**

1. In the on-policy setting, GAE with $$\lambda=0.9$$ is a recommendable default choice for majority of environments. 

## Training Setup

**Design decisions:** number of parallel environments, number of transitions gathered in each iteration, number of passes over the data, mini batch size, how the data is split into mini batches

**Takeaways:**

1. Go over experience multiple times (allows to quickly find an approximately optimal policy).

2. Shuffle individual transitions before assigning them to mini batches and recompute advantages once per data pass (PPO-specific advice).

3. For faster wall-clock time training use many parallel environments and increase the batch size (both might hurt the sample complexity).

4. Tune the number of transitions in each iteration if possible.

**Comments:** This paragraph has brought my attention to the fact that naively increasing number of parallel environments does not automatically lead to faster training. Over last few months I have noticed some TF-Agents users actually complaining that it can actually be harmful ([here](https://github.com/tensorflow/agents/issues/336)). This is definitely something I would like to investigate further.

## Timesteps Handling

**Design decisions:** discount factor $$\gamma$$, frame skip, episode termination handling.

**Takeaways:**

1. Discount factor $$\gamma$$ turns out to be pretty crucial for the performance and should be individually tuned for every environment (with the default value of 0.99). 

2. Frame skipping (the number of frames an action is repeated before a new action is selected) can also help in some cases. 

3. For large step limits, there is no need to handle environment step limits. 

**Comments:** One of the problems with the conventional RL benchmarks is the fact, that we test agents in environments where episodes have fixed number of steps. However, this information is hardly ever contained in the agent's state and thus violates the Markovian assumption of MPDs. Why assuming infinite time horizon but training for finite time horizon could be a problem? Think about a tie in the basketball game. Players may employ very different strategies if the draw happens in the middle of the game or just before the final whistle. But agents usually do not take this crucial information into consideration. We can treat those problematic, <em>abandoned</em> last steps either as terminal steps, or assume that we do not know what would happen if the episode was not terminated. 

The experiments show that it is not important how we handle <em>abandoned</em> episodes, as long as the number of time steps is large. So I guess it is fair to assume, that for more complex environments and/or smaller number of time steps in an episode, this statement may not hold true anymore. 

## Optimizers

**Design decisions:** choice of different gradient-based optimizers (e.g. Adam, RMSprop) and their hyperparameters on the performance of the benchmark environments.

**Takeaway:**

1. As a default, use Adam optimizer with momentum $$\beta_{1}=0.9$$ . Start with the default learning rate $$0.0003$$, but be sure to adjust it to your problem. 

2. Linear decaying may slightly improve the performance. 

**Comments:** This is pretty much consistent with what we already know about the most popular optimizers. Adam is pretty much always a safe default. And regarding the default learning rate, well, Andrej Karpathy pretty much said it [four years ago](https://twitter.com/karpathy/status/801621764144971776?s=20).

## Regularization

**Design decisions:** choice of different policy regularisers (e.g. entropy, KL divergence between action distribution and a unit Gaussian) and their hyperparameters on the performance of the benchmark environments.

**Takeaway:**

Well, here the results of the experiments were not very spectacular. Any form of regularization (be it entropy, Kullback-Leibler divergence between the unit Gaussian and the policy action distribution, reverse KL divergence between the target and behavioural policy) does not help much. But we have to keep in mind, that all the agents were trained with the PPO loss, which already enforces the trust region. This means that it already incorporates a certain form of regularization.

# Conclusions

**I am really happy that there are people who are looking into "practical" aspects of deep RL such as reproducibility or good engineering practices**. It's great that researchers are looking for some general rules which may hold true for majority of problems and could be used to accelerate deep RL prototyping. **I would like to conclude this write-up with one, crucial critique of the publication**: the authors have been conducting this study for a very limited set of benchmark environments. All of them assume state and observation space to be 1D vectors and they all can be "solved" using MLP-based networks. I wonder if some of those learnings are also valid for more complex tasks, be it multi-agent settings or agents which deal with large state spaces (e.g work with multimodal camera input)?
