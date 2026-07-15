# 强化学习（1）
## 前情提要
最近Lab里面一直在做Robot的Policy调参，+1又一直在念叨什么 PPO、GRPO、SAO 各种新奇名词。但是本大一萌新本科生根本连Reinforcement Learning, RL的基本概念都不知道，故开始从头恶补ing~

## 基本术语 Terminologies
- State：状态
- Action：动作
- Policy：策略$\pi$（在状态$s$下选$a$的条件分布）

---

- Reward：奖励
- State Transition：当前状态$s$由于进行了动作$a$，使状态发生的变化（由环境决定）

## RL的随机性 Randomness
1. 策略$\pi$的随机抽样导致的随机性
2. 环境在状态变化时的随机性

## 回报 Return
1. 回报是未来所有奖励的累积
2. 折扣回报是未来所有奖励乘以折扣权重

### 公式定义
1. Return (accumulative future reward)
$$U_t = R_t + R_{t+1} + \dots + R_T$$
- Given that future reward is less valuable than present reward
- Let $\gamma$ be the discount rate of the reward

2. Discounted Return (accumulative discounted future reward)
$$U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots$$

---

## 价值函数 Value Function
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