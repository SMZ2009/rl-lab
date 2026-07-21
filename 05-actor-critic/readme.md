# Chapter 5 Actor-Critic 架构

# 1. REINFORCE 回顾

参数的更新规则：

\[
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \, G_t \right]
\]

\[
\theta \leftarrow \theta + \alpha \, \gamma^t \, G_t \, \nabla_\theta \log \pi_\theta(a_t|s_t)
\]

\[
G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}
\]

目的是尽可能让 \(J(\theta)\) 期望累积回报最大

\[
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ G(\tau) \right]
\]
\[
G(\tau) = \sum_{t=0}^{T-1} \gamma^t r_{t+1}
\]

但是存在：**高方差**、**无法在线更新**等缺点

\[
    Var(G_t) = \mathbb{E}[Var(G_t|S_t)] + Var[\mathbb{E}(G_t|S_t)]
    \]

第一项代表给定状态下的内在随机性（策略随机、环境转移噪声），而第二项代表不同状态间的回报差异。

能不能消除环境的不同状态本身的影响，专注学习内在的策略呢？

# 2. Actor 与 Critic

我们把动作策略$\pi_{\theta}(a|s)$定义为**演员 Actor**，把价值函数定义为**评论家 Critic**。

如果我们引入一个只与状态有关的函数，如状态价值函数$V_\pi(s)$（衡量当前的状态有多好）。然后用动作价值函数$Q_\pi(s_t, a_t)$（衡量在当前的状态下做动作a有多好）减去状态价值函数，是不是就能够衡量动作策略的相对进步有多大？

形象地说，把动作价值函数类比成体操运动员的比赛得分，把状态价值函数理解为动作的难度分，将他们相减就能得到运动员动作的相对好坏。

所以我们引入**Advantage**来衡量相对进步

\[ A^{\pi}(s_t,a_t) = Q^{\pi}(s_t,a_t) - V^{\pi}(s_t)\]

而
$$Q_\pi(s_t, a_t) = \mathbb{E}_{\pi}[U_t | S_t = s_t, A_t = a_t] = \mathbb{E}_{\pi}[r_t + \gamma V_{\pi}(s_{t+1})| S_t = s_t, A_t = a_t]$$

所以

$$A^{\pi}(s_t,a_t) = r_t + \gamma V_{\pi}(s_{t+1}) - V_{\pi}(s_t)$$

刚好就是 TD 误差的形式

---

你可能会说，Chapter 2 我们都已经介绍了 TD 算法，怎么中间绕了一大圈又回来了呢？

一开始的 TD 算法是去用来更新价值函数的（价值学习策略），引出了 SARSA 与 Q-learning（DQN）。但是他们都是去学习动作带来的价值，并没有直接去学习策略。

- 价值学习无法处理连续动作：本质是通过贪心策略或$\argmax$来进行更新，只能处理离散动作，不适用于robot等场景。

- 价值学习输出的是一个确定的策略（目的是让动作价值函数最大化）：但是在实际的博弈策略中，输出的策略必须是随机的。

所以我们在这里是用$A_t$ Critic 来更新$\theta$也就是 Actor，而$A_t$的更新我们复用了当时价值函数的 TD 方法。

---

由于状态价值函数$V_\pi(s)$不改变$J(\theta)$的期望，可以改写成

\[
\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \, A_t \right]
\]

$A_t$可以通过 TD 算法进行更新

$$A^{\pi}(s_t,a_t) \sim r_t + \gamma V_{\omega}(s_{t+1}) - V_{\omega}(s_t)$$

$\pi$代表真实的动作策略，$\omega$代表用神经网络逼近的策略

$\pi$代表 Actor 网络，$\omega$代表 Critic 网络

# 3. Actor-Critic 算法

每一步 \( s_t, a_t, r_t, s_{t+1} \) 收集到后即开始在线学习

计算 TD error：

\[\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t) \]

价值网络更新：

\[w \leftarrow w + \alpha_w \, \delta_t \, \nabla_w V_w(s_t)\]

策略网络更新：

\[\theta \leftarrow \theta + \alpha_\theta \, \delta_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\]