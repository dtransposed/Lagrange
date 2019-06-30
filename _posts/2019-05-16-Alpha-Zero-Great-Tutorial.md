---
layout: post
title: "The Ultimate AlphaGoZero Tutorial "
author: "Damian Bogunowicz"
categories: blog
tags: [reinforcement learning, multi-armed bandits, artificial intelligence, game theory]
image: alpha-zero-one.png
---









## Upper Confidence Bound

One of the simplest policies for making decisions based on action values estimates is greedy action selection. 


$$
A_t = \underset{a}{\mathrm{argmax}}(Q_t(a))
$$


This means, that in order to choose an action $$A_t$$ we compute an estimated value of all the possible actions and pick the one which has the highest estimate. This greedy behavior completely ignores the possibilities that some actions may be inferior in the short run but ultimately superior in the longer perspective. We can fix this shortcoming by deciding that sometimes (the "sometimes" is parametrized by some probability $$\epsilon$$) we simply ignore the greedy policy and pick a random action. This is the simplest way to enforce some kind of exploration in this exploitation-oriented algorithm. Such a policy is known as epsilon-greedy.

Even though epsilon-greedy policy does help us to explore the non-greedy actions, it does not model the related uncertainty. It would be much more clever to make decisions by also taking into account our belief, that some action value estimates are much more reliable i.e. they are closer to some actual, unknown action value $$q_{*}(a)$$ then others. Even though we do not know  $$q_{*}(a)$$ (it is actually our goal to estimate it), we can use the notion of the difference between the desired $$q_{*}(a)$$ and currently available $$Q_t(a)$$. This relationship is described by the Hoeffding's Inequality. 

Imagine we have $$t$$  independent, identically distributed random variables, bounded between 0 and 1. We may expect that their average is somewhat close to the expected value. The Hoeffding's inequality precisely quantifies the relationship between $$\bar{X_t}$$ and $$\mathbb{E}[X]$$.


$$
Pr[|\bar{X_t}-\mathbb{E}[X]|>m] \leq e^{-2tm^2}
$$


The equation states that the probability that the sample mean will differ from its expected value by some threshold $$m$$ decreases exponentially with increasing sample size $$t$$ and increasing threshold $$m$$. In other words, if we want to increase the probability that our estimate improves, we should either collect more samples or agree to tolerate higher possible deviation.

In our multi-armed bandit problem we want to make sure that our estimated action value is not far from the real action value. We can express the threshold $$m$$ as a function of an action and call it $$U_t(a)$$. This value is known as an upper confidence bound. Now, we can apply the Hoeffding's inequality and state:


$$
Pr[|Q_t(a)-q_*(a)|> U_t(a)] \leq e^{-2tU_t(a)^2}
$$


 Let's denote this probability as some very small number $$l$$ and transform this expression.


$$
l = e^{-2NU_t(a)^2}
$$


$$
U_t(a) = \sqrt{-\log{l}/2N_t(a)}
$$

We can make $$l$$ dependent on the number of iterations (let's say $$l = t^{-4}$$)  and rewrite the equation.


$$
U_t(a) = \sqrt{-\log{t^{-4}}/2N_t(a)} = C\sqrt{\log{t}/{N_t(a)}}
$$


Finally, we can write down the UCB policy:


$$
UCB(a) = \underset{a}{\mathrm{argmax}}(Q_t(a) +  C\sqrt{\log{t}/{N_t(a)}})
$$


The choice of $$l$$ is in practice reflected by the parameter $$C$$ in front of the square root. It quantifies the degree of exploration. With the large $$C$$ we obtain greater numerator value of the square root and make the uncertainty expression more significant with respect to the overall score.  However, ultimately $$U_t(a)$$ is bound to decay, since the numerator ($$N_t(a)$$ - number of times we have chosen action $$a$$) increases with the higher rate then the numerator. 

We may write down a pseudo-code for a simple, multi-armed bandit algorithm (UCB1) :

```
Initialize a Bandit with:

p # arm pull number
a # possible actions a 
c # degree of exploration

Q(a) = 0	# action value estimates
N(a) = 0	# number of times the actions are selected
t = 0

for pull in range(p):
	t = t + 1
	# choose an action according to UCB formulation
	A = argmax(Q(a) + c*sqrt(ln(t)/N(a)))		
    
	R = Bandit(A)			# receive a reward from the environment
	N(A) = N(A) + 1			# update the action selection count	
	Q(A) = Q(A) + (R-Q(A)/N(A)	# update action value estimate
	
	
```

Finally, I have implemented UCB1 algorithm in Python: you can find the code here. I have tested it for a bandit with ten arms and 1000 runs.  Note, that that my implementation makes algorithm do some random exploration, it is not pure vanilla UCB1. 

The following diagrams illustrate how the algorithm incrementally estimates the probability of receiving a reward from each of the arms for one single run.

<img src="/assets/7/giphy.gif"/> 

<img src="/assets/7/true_probability.png" width="500"/> 

<em> With every iteration, the UCB1 improves the estimate of the true reward probability for each available action. The orange parts of the bar chart decrease continuously over iterations, while the blue parts get close to the true distribution.   </em>

Next experiment involves running the algorithm 1000 times and computing the percentage of optimal actions taken for each iteration of UCB1.

<img src="/assets/7/UCB-1-results.png" width="500"/> 

<em> Over time, the algorithm succesfully learns to select optimal actions. </em>



##  MCST & UCT

Monte Carlo Search Tree (MCST)  is a heuristic search algorithm which helps to make optimal choices in decision processes (e.g games). This means it can be theoretically applied to any domain that can be described as a (state, action) tuple.

In principle, the MCST algorithm consists of four steps:

1. Selection: the algorithm starts at the root node R (initial state of the game) and traverses the decision tree (so it revisits the previously "seen" states ) according to the current policy until it reaches some leaf node L. If node L:

   ​	a) is a terminal node (final state of the game) - skip to step 4.

   ​	b) otherwise, node L has some previously unexplored children - continue with step 2.

2. Expansion: If node L is not terminal, we expand one of the child nodes of L - let's call this child node C.

3. Rollout: Starting from node C, we let the algorithm simply continue playing on it's own according to some policy (e.g random policy) until we reach a terminal node.

4. Once we reach a terminal node, we write down the final score. Then, all the nodes visited in this iteration (starting with C, through all the nodes involved in selection step, up to the root node R) are updated. This means we update their value and number of times each of the nodes has been visited.

UCT (Upper Confidence Bound for Search Trees) combines the concept of MCST and UCB. This means introducing a small change to the rudimentary tree search: in selection phase, for every parent node the algorithm evaluates its child nodes using UCB formulation:


$$
UCT (j) =\bar{X}_j + C\sqrt{\log(n_p)/(n_j)}
$$


Where $$\bar{X}_j$$ is an average value of the node, $$C_p$$ is some positive constant (responsible for exploration-exploitation trade-off), $$n_p$$ is the number of times the parent node has been visited and $$n_j$$ is the number of times the node $$j$$ has been visited.

The child node with higher UCB score gets selected. 

Once we reach the desired number of iteration, 

In general MCTS would be preferred over other tree search methods because of its:



Sources:

[MCTS research hub](http://mcts.ai/about/index.html) - excellent starting point for getting familiar with the algorithm

[UCT video tutorial by John Levine](https://www.youtube.com/watch?v=UXW2yZndl7U) - short and clear explanation of the algorithm, together with a worked example. 



UCT (Upper Confidence bounds applied to Trees)

https://www.youtube.com/watch?v=UXW2yZndl7U

David Silver's UCB video:
...
https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html#ucb1
Inspiration:
http://www.depthfirstlearning.com/2018/AlphaGoZero

Learning minmax

<https://www.youtube.com/watch?v=fInYh90YMJU>



Shortly about MC:

[http://jhamrick.github.io/quals/planning%20and%20decision%20making/2015/12/16/Browne2012.html](http://jhamrick.github.io/quals/planning and decision making/2015/12/16/Browne2012.html)

Polak UCB

<https://github.com/dloranc/reinforcement-learning-an-introduction/blob/master/01_multi_arm_bandits/05_ucb.py>

quick alpha zero:

<https://tmoer.github.io/AlphaZero/>

<https://jeremykun.com/2013/10/28/optimism-in-the-face-of-uncertainty-the-ucb1-algorithm/>



Sources:

Creating AlphaZero:
https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188

AlphaZero Cheatsheet:
https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0

AlphaZero Stanford:
https://web.stanford.edu/~surag/posts/alphazero.html