# Chapter 0 强化学习基础

接下来，我将开启我的强化学习算法分享之路。

老实说，现在世面上的强化学习算法并不是很“小白友好”，能够零基础入门。所以作为一名在Lab里面一直调整Robot的Policy，并且被+1轰炸PPO、GRPO、SAO各种新奇名词的准大二学生，我想要自己动手丰衣足食。

所以，我将用最直白、最通俗易懂、最不绕弯子的方法来介绍强化学习。

本期聚焦 **基本术语** **回报** 与 **价值函数**，希望能让大家在接触各个系列与流派的算法之前先有一个概念的基础。

# 1. 基本术语 Terminologies

- 智能体agent在某一状态state下做出某一动作action。

- 智能体在状态s下选择动作a的条件分布就是它的动作策略policy。

- 智能体选择了某一个动作a之后，会得到奖励reward（可正可负）。

- 当前状态s由于智能体做了动作a之后，会变化成s'状态，这就是状态转移state transition，由环境决定。

所以说，在强化学习的过程中，随机性是怎么出现的呢？

首先是智能体选择哪一个动作action，会根据策略 $\pi$ 进行随机抽样。

还有是状态转移函数 $\varepsilon$ 的随机性，因为它会受环境变化影响。

# 2. 回报 Return

- 回报是未来所有奖励的累积
   
$$U_t = R_t + R_{t+1} + \dots + R_T$$

- 折扣回报是未来所有奖励乘以折扣权重

$$U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots$$

怎么理解？未来的回报肯定没有当下的回报值钱，所以说有一个折扣权重 $\gamma$


# 3. 价值函数 Value Function
1. 动作价值函数$Q(\pi)$：衡量在状态$s$下做动作$a$，并按照$\pi$这个策略一直进行下去得到的回报的均值
2. 最优动作价值函数$q^*$：衡量在状态$s$下做动作$a$，以一定的策略能够得到的最大的回报的均值
3. 状态价值函数：在状态$s$下，由策略$\pi$随机挑选并按照$\pi$这个策略一直进行下去得到的回报的均值
4. 对状态价值函数在每一个不同的状态$s$下求期望，也就是在不同状态下依靠策略$\pi$得到的回报均值，能够反映策略$\pi$有多好

### 公式定义
1. Action-Value Function for policy
$$Q_\pi(s_t, a_t) = \mathbb{E}[U_t | S_t = s_t, A_t = a_t]$$
- Optimal action-value function
$$Q^*_\pi = \max_\pi Q_\pi(s_t, a_t)$$
- How good it is for an agent to pick an action $a$ while being in a state $s$

2. State-Value Function
$$V_\pi(s) = \mathbb{E}_\pi[U_t | S_t = s_t]$$
- How good the situation is while being in a state of $s$
- $\mathbb{E}_s[V_\pi(s)]$ evaluates how good your action policy is