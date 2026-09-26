# Lesson 2 — Learning from Experience: Q-learning

> **Goal:** learn the optimal action values $Q^*$ **without** a model of the world,
> just by playing, and use them to control a marine in StarCraft II.
>
> **You will learn:** sample averages, temporal-difference (TD) learning, the
> Q-learning update, exploration vs exploitation, $\varepsilon$-greedy, designing
> states for a real game, state aliasing.
>
> **Time:** about 2–3 hours. **StarCraft II needed:** yes (from section 7).

---

## 1. Where we are

In [Lesson 1](../01_rl_basics/) we computed the optimal policy with value iteration.
It needed the model $p(s', r \mid s, a)$: a full list of what can happen after every
action. StarCraft II gives us no such list. All we can do is:

```python
next_state, reward, done = env.step(action)
```

We get **one sample** of what can happen, not the probabilities. Can we still find
the optimal policy? Yes, and the idea is surprisingly simple.

---

## 2. Idea 1: averages can be learned one sample at a time

Suppose you want the average of numbers $x_1, x_2, x_3, \dots$ that arrive one by one,
and you do not want to store them all. Let $\bar{x}_n$ be the average of the first $n$.
A little algebra gives:

$$
\bar{x}_n = \bar{x}_{n-1} + \frac{1}{n}\left( x_n - \bar{x}_{n-1} \right) \tag{1}
$$

*Example.* Numbers 4, 8, 6. Start with $\bar{x}_1 = 4$.
Then $\bar{x}_2 = 4 + \frac12(8 - 4) = 6$, and $\bar{x}_3 = 6 + \frac13(6 - 6) = 6$. ✔

Read Eq. 1 as:

$$
\text{new estimate} = \text{old estimate} + \text{step size} \times \left( \text{target} - \text{old estimate} \right)
$$

The part in brackets is the **error**: how wrong our estimate was on this sample.
We move the estimate a little bit toward the new sample. In RL we usually replace
$\frac{1}{n}$ by a small constant $\alpha$ (the **learning rate**), which gives more
weight to recent samples. That is useful because, as we will see, our targets
improve over time.

---

## 3. Idea 2: the Bellman equation is an average we can sample

From Lesson 1 (Eq. 7 and Eq. 8), the optimal action values satisfy:

$$
Q^*(s, a) = \sum_{s', r} p(s', r \mid s, a) \left[\, r + \gamma \max_{a'} Q^*(s', a') \,\right] \tag{2}
$$

The right-hand side is an **expected value** (a probability-weighted average) of the
quantity in brackets. We cannot compute the sum without $p$, but every time we play
action $a$ in state $s$, the game hands us one sample $(r, s')$ drawn from exactly
that distribution $p(s', r \mid s, a)$.

So we can average the samples with Eq. 1. That is all Q-learning is.

---

## 4. The Q-learning update

After each step $(s, a, r, s')$, compute the **TD target** (our one-sample guess of
the right-hand side of Eq. 2) and the **TD error** (how far off we were):

$$
y = r + \gamma \max_{a'} Q(s', a'), \qquad \delta = y - Q(s, a) \tag{3}
$$

Then move $Q(s,a)$ a small step toward the target, exactly like Eq. 1:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha\, \delta \tag{4}
$$

If $s'$ is a **terminal** state, nothing comes after it, so $y = r$.

In [`q_learning.py`](q_learning.py):

```python
future = 0.0 if terminal else np.max(self.Q[next_state])
td_target = reward + self.gamma * future                      # Eq. 3
td_error = td_target - self.Q[state][action]                  # Eq. 3
self.Q[state][action] += self.alpha * td_error                # Eq. 4
```

**Learning a guess from a guess.** Notice that the target $y$ uses $Q(s', \cdot)$,
which is itself only an estimate. This is called **bootstrapping**. At first all
estimates are 0 and only the step that reaches the beacon gets a real signal
($r = 1$). On later episodes that value flows backwards, one step at a time, to the
states before it. Section 6 shows this happening.

The name **temporal-difference (TD)** learning comes from the TD error $\delta$: it is
the difference between two estimates at successive time steps.

---

## 5. Exploration vs exploitation

If the agent always picks the action with the highest current $Q$ (**exploitation**),
it gets stuck: early on, all values are 0, and it will never try the actions that
might turn out to be better. It must sometimes try other actions (**exploration**).

The simplest fix is **$\varepsilon$-greedy**:

$$
a =
\begin{cases}
\text{a random action} & \text{with probability } \varepsilon \\
\arg\max_{a} Q(s, a) & \text{with probability } 1 - \varepsilon
\end{cases}
\tag{5}
$$

We start with $\varepsilon = 1$ (explore all the time) and lower it linearly to
$0.05$ over the first 60% of training (`linear_schedule` in the code). When several
actions tie for the best value, we pick one of them at random.

**Off-policy.** The agent *behaves* $\varepsilon$-greedily, but the $\max$ in Eq. 3
means it *learns about* the greedy policy. Learning about one policy while following
another is called **off-policy** learning. It is why Q-learning can explore freely
and still learn the optimal values.

---

## 6. The whole algorithm, and a first test in GridWorld

```
Initialise Q(s, a) = 0 for all s, a
For each episode:
    s = env.reset()
    Until the episode ends:
        choose a with ε-greedy (Eq. 5)
        s', r, done = env.step(a)
        δ = r + γ max_a' Q(s', a') − Q(s, a)     (Eq. 3; just r if s' is terminal)
        Q(s, a) ← Q(s, a) + α δ                   (Eq. 4)
        s = s'
```

▶ **Run it** (a few seconds, no StarCraft needed):

```bash
python 02_q_learning/q_learning.py --env gridworld
```

```
episode   50 | epsilon 0.74 | avg reward   0.68 | avg steps   32.1
episode  100 | epsilon 0.48 | avg reward   1.00 | avg steps   15.9
episode  150 | epsilon 0.21 | avg reward   1.00 | avg steps    8.9
episode  200 | epsilon 0.05 | avg reward   1.00 | avg steps    5.7
episode  250 | epsilon 0.05 | avg reward   1.00 | avg steps    5.3
episode  300 | epsilon 0.05 | avg reward   1.00 | avg steps    5.2

Learned greedy policy (compare with Lesson 1's value iteration):
v v > v <
> v # v v
> v # v <
v v v v v
> > > > B
```

![GridWorld learning curve](results/gridworld_learning_curve.png)

The number of steps drops from about 32 (mostly random) to about 5, the length of the
shortest path. The last few episodes are slightly above 5 because $\varepsilon = 0.05$
still makes a random move now and then.

**Compare with value iteration.** The policy is not identical to Lesson 1's. Along
the paths the agent actually uses, it is optimal. But look at the top-right corner:
`<` means "go left", which is not the shortest way. The marine almost never visits
that corner, so the agent never learned much about it. **Q-learning only knows about
what it has experienced.** Value iteration, with the full model, knew everything.

---

## 7. Moving to StarCraft II: designing the state

The PySC2 game gives us images with many layers and hundreds of possible actions.
A table cannot handle that, so the wrapper [`envs/move_to_beacon.py`](../envs/move_to_beacon.py)
builds a small state and action set for us. These are *design choices*, and they
matter as much as the algorithm.

**Actions.** 8 directions. Each action orders the marine to walk 8 pixels that way.

**State.** Where is the beacon *relative to* the marine? We take the offset
(beacon − marine) in screen pixels, divide it into cells of 4 × 4 pixels, and clip it
to at most 8 cells in each direction. That gives $17 \times 17 = 289$ states, and
the Q-table has $289 \times 8 = 2312$ numbers.

Using the **relative** position is a good trick: "the beacon is 3 cells to the
right" needs the same action no matter where on the map the marine stands, so the
agent learns once and reuses it everywhere.

**Reward.** +1 every time the marine touches the beacon. The beacon then jumps to a
random new place. An episode lasts 120 game seconds (about 240 steps).

### A lesson from building this: state aliasing

The first version of the wrapper used **8-pixel** cells. Even a hand-written
"always walk toward the beacon" policy scored almost nothing. Printing the positions
showed why: the marine stood 4 pixels from the beacon, which rounds to the cell
"offset (0, 0)", the same state as *standing on* the beacon. The state could not tell
"I am on the beacon" apart from "the beacon is right next to me", so no policy could
choose correctly. When two different situations that need different actions look
like the same state, it is called **state aliasing**. With 4-pixel cells the
scripted policy scores about 23 per episode.

### Terminal states in MoveToBeacon

The game episode does not end when the beacon is reached; the beacon just moves.
But the next beacon position is random and has nothing to do with the action we
took. So for learning we treat "reached the beacon" as the end of a sub-task:
$y = r = 1$ with no future term (`terminal_on_reward=True` in the code).

The 120-second time limit is **not** a terminal state. The marine did nothing wrong
there; the clock just ran out. We do not cut the future off at a time-out.

---

## 8. Q-learning in StarCraft II

▶ **Run it** (about 6–7 minutes; the game window opens but the agent plays at full speed):

```bash
python 02_q_learning/q_learning.py --env sc2 --episodes 300
```

```
episode   10 | epsilon 0.95 | avg reward   0.30 | avg steps  239.0
episode   50 | epsilon 0.74 | avg reward   2.80 | avg steps  239.0
episode  100 | epsilon 0.48 | avg reward   9.70 | avg steps  239.0
episode  150 | epsilon 0.21 | avg reward  15.80 | avg steps  239.0
episode  200 | epsilon 0.05 | avg reward  20.30 | avg steps  239.0
episode  250 | epsilon 0.05 | avg reward  19.50 | avg steps  239.0
episode  300 | epsilon 0.05 | avg reward  19.50 | avg steps  239.0
```

(Your numbers will differ a little: the beacon positions are random.)

![MoveToBeacon learning curve](results/sc2_learning_curve.png)

The marine starts at about 1 beacon per episode (random) and reaches about 20, close
to the hand-written "walk straight to the beacon" policy (about 23). Nobody told the
agent where to go. It learned it from the +1 rewards alone.

**Reading the policy map.**

![MoveToBeacon policy map](results/sc2_policy.png)

Each cell is a **state**: where the beacon is relative to the marine. The red star in
the middle means "the beacon is right here". The cell three to the right of the star
means "the beacon is 3 cells to my right", and its arrow says what the agent does
there. The arrows point **away from the centre**, which is correct: if the beacon is
to your right, walk right. The colour is $\max_a Q(s, a)$: bright near the centre
(the reward is close), dark far away (the reward is many discounted steps away, as
$\gamma^{d}$ in Lesson 1).

Near the edges the arrows are messier. Those states are visited less often, the same
effect as the top-right corner of GridWorld.

Why not 23? With $\varepsilon = 0.05$ the agent still takes a random step now and
then, and the state is coarse (4-pixel cells), so it sometimes zig-zags. Experiment 4
in section 9 explores the state size.

▶ **Watch the trained marine** at human speed:

```bash
python 02_q_learning/watch.py
python 02_q_learning/watch.py --random     # for comparison
```

---

## 9. Experiments

Change one thing at a time. Predict first, then run.

1. **Learning rate.** `--alpha 0.5` and `--alpha 0.01`. Which learns faster at first?
   Which ends up more stable?
2. **Discount.** `--gamma 0.5` and `--gamma 0.99` in GridWorld. Look at the policy map
   in `results/gridworld_policy.png`.
3. **Always explore / never explore.** In `train(...)`, set `eps_start=eps_end=1.0`,
   then `eps_start=eps_end=0.0`. What happens in each case, and why?
4. **State aliasing.** Create the environment with `MoveToBeaconEnv(cell_pixels=8, radius=4)`
   and train again. Does the score match section 7's story?

---

## 10. The limit of tables

Our table has one number per (state, action). That worked because we squeezed the
game into 289 states by hand. For real StarCraft screens, $84 \times 84$ images with
many layers, the number of possible states is astronomically large; a table is
impossible, and the agent could never visit each state even once.

We need a function that **generalises**: something that, having learned about one
state, can guess the value of similar states it has never seen. A neural network can
do that. Replacing the table with a network gives **Deep Q-Networks (DQN)**, the
topic of Lesson 3.

---

## 11. Exercises

1. Starting with all $Q = 0$, $\alpha = 0.1$, $\gamma = 0.9$: the agent is in state $s$,
   takes action $a$, gets $r = 1$ and reaches a terminal state. What is $Q(s,a)$ after
   the update? And after the same thing happens a second time?
2. Now the agent is in $s_0$, takes $a_0$, gets $r = 0$ and lands in the $s$ from
   exercise 1 (after the second update). All other values are still 0. What is the
   new $Q(s_0, a_0)$?
3. In exercise 2, the value "flowed" from $s$ back to $s_0$. How many episodes are
   needed, at the very least, for any value to reach the start state of GridWorld,
   5 steps from the beacon?
4. Why does Eq. 3 use $\max_{a'} Q(s', a')$ and not the $Q$ of the action the agent
   actually takes next?

<details>
<summary>Answers</summary>

1. First: $\delta = 1 - 0 = 1$, so $Q = 0 + 0.1 \times 1 = 0.1$.
   Second: $\delta = 1 - 0.1 = 0.9$, so $Q = 0.1 + 0.1 \times 0.9 = 0.19$.
   Each time, $Q$ moves 10% of the remaining way toward 1.
2. $y = 0 + 0.9 \times 0.19 = 0.171$, $\delta = 0.171$, so $Q(s_0,a_0) = 0.0171$.
3. At least 5. In each episode, value can move back by one step along the path
   (the update happens when you *leave* a state, using the next state's value as it
   is at that moment). This is one reason Q-learning can be slow when rewards are
   rare.
4. Because we want $Q^*$, the value of acting **optimally** from $s'$ onwards (Eq. 2),
   not the value of our exploring behaviour. Using the action actually taken gives a
   different algorithm called **SARSA**, which learns the value of the
   $\varepsilon$-greedy policy itself (on-policy).

</details>

---

## Summary

| Idea | Equation | In one line |
|---|---|---|
| incremental average | Eq. 1 | estimate += step · (sample − estimate) |
| TD target / TD error | Eq. 3 | $y = r + \gamma \max Q(s',\cdot)$, $\delta = y - Q(s,a)$ |
| Q-learning update | Eq. 4 | $Q(s,a) \mathrel{+}= \alpha \delta$ |
| $\varepsilon$-greedy | Eq. 5 | random with probability $\varepsilon$, else best |

**For the curious.** Tabular Q-learning is proven to converge to $Q^*$ if every
(state, action) pair is tried infinitely often and the learning rate shrinks in the
right way ($\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$). See
Sutton & Barto, chapter 6.5.

**Further reading:** Sutton & Barto, chapter 6 (Temporal-difference learning).

**Previous:** [← Lesson 1](../01_rl_basics/) · **Next:** Lesson 3 — Deep Q-networks (coming soon)
