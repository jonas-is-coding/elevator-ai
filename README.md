# 🏗️ Elevator-AI: Reinforcement Learning for Elevator Dispatch Optimization

> Comparing a classical Destination Dispatching algorithm against a PPO-trained reinforcement learning agent in a custom-built Gymnasium simulation environment.

**Stack:** Python · Gymnasium · stable-baselines3 · FastAPI *(in progress)* · Next.js *(in progress)*

---

## Results at a Glance

|  | Classic Agent | PPO Agent |
|---|---|---|
| **Mean Avg Reward** | -0.67 | **+0.14** |
| **Mean Dropoffs / Episode** | 2,664 | **2,855** (+7%) |
| **Mean Avg Wait (steps)** | 600.61 | **93.06** (6× faster) |

*Evaluated over 1,000 independent episodes of 10,000 steps each.*

---

## Project Structure

```
elevator-ai/
├── environment/
│   ├── building.py          # Core simulation entities
│   ├── elevator_env.py      # Gymnasium environment
│   └── traffic_patterns.py  # Probabilistic passenger spawning
├── agents/
│   ├── classic_agent.py     # Destination Dispatching baseline
│   └── ppo_agent.py         # PPO model definition
├── training/
│   └── train.py             # Training script
├── tests/
│   ├── test_classic.py      # Classic agent evaluation
│   └── test_ppo.py          # PPO agent evaluation
└── README.md
```

---

## Getting Started

```bash
pip install gymnasium stable-baselines3 numpy
```

**Train the PPO agent:**
```bash
cd training
python train.py
```

**Evaluate both agents:**
```bash
cd tests
python test_classic.py
python test_ppo.py
```

---

## Roadmap

- [x] Simulation environment (Phase 1 + Phase 2)
- [x] Classical Destination Dispatching baseline
- [x] PPO agent training and evaluation
- [ ] Realistic elevator speed simulation *(in progress)*
- [ ] FastAPI inference server
- [ ] Next.js visualization frontend

---

---

# 📄 Research Paper

## Reinforcement Learning for Elevator Dispatch: A Comparative Study of PPO Against Classical Destination Dispatching

**Jonas Brahmst** · Independent Research Project · 2025

---

### Abstract

This paper investigates whether a Proximal Policy Optimization (PPO) reinforcement learning agent can outperform a classical Destination Dispatching algorithm in a simulated multi-elevator environment. A custom Gymnasium environment was designed to model a 20-floor building with 4 elevators and probabilistic passenger traffic. After extensive reward engineering and architectural iteration, the PPO agent achieved a mean average reward of **+0.14** per step compared to **-0.67** for the classical baseline — a significant improvement driven primarily by a **6× reduction in average passenger wait time** (93 vs 601 steps).

---

### 1. Introduction

Elevator dispatch algorithms are a well-studied problem in operations research. Modern buildings typically rely on rule-based systems such as SCAN or Destination Dispatching — deterministic algorithms that assign elevators to floors based on proximity and direction. While effective in predictable traffic conditions, such systems cannot adapt to learned patterns.

This project explores whether a reinforcement learning agent trained via PPO can discover a superior dispatch policy from experience alone, without any hand-coded routing logic. The comparison is designed to be rigorous: both agents operate in identical environments under identical traffic conditions, and performance is averaged over 1,000 independent evaluation episodes.

---

### 2. Environment Design

#### 2.1 Building Simulation

The simulation models a 20-floor building with 4 independent elevators. Each elevator maintains a list of target floors representing passengers currently aboard. Each floor tracks whether passengers are waiting to travel up or down, along with how long they have been waiting.

```
Building:    20 floors, 4 elevators
Max steps:   10,000 per episode
Time model:  1 step ≈ 1 elevator floor traversal
```

A deliberate design decision was made to model elevator call buttons as boolean flags (`waitingUp`, `waitingDown`) rather than integer counters. This reflects reality: a floor call button does not convey how many people are waiting — only that someone is. This keeps the simulation grounded in the information a real elevator system would actually have access to.

#### 2.2 Traffic Patterns

Passenger spawning is probabilistic with time-of-day modifiers:

| Period | Floors affected | Multiplier | Direction bias |
|---|---|---|---|
| Off-peak | All | 1× | Random |
| Morning Rush (07–10) | Floor 0 | 3× | Up (100%) |
| Morning Rush (07–10) | Floors 1–19 | 0.3× | Random |
| Evening Rush (16–20) | Floors 1–19 | 2× | Down (80%) |

Base probability was set to `0.02` per floor per step, yielding approximately 0.4 passengers per step across the building — a realistic load for 4 elevators.

#### 2.3 Phase 2: Passengers in Elevators

A critical design decision was to implement a full two-phase passenger model:

- **Phase 1:** Passenger waits on floor with a directional call
- **Phase 2:** Passenger boards elevator, a random destination floor is assigned, passenger travels to destination

Upon boarding, destination floors are sampled uniformly from valid floors in the travel direction:

```python
# Boarding going up from floor N:
target = random.randint(floor.number + 1, num_floors - 1)

# Boarding going down from floor N:
target = random.randint(0, floor.number - 1)
```

This models the real-world scenario where the elevator system does not know a passenger's destination until they board and press a button.

#### 2.4 Observation Space

The observation vector has **92 values**:

| Component | Count | Description |
|---|---|---|
| Elevator position | 4 | Current floor (0–19) |
| Elevator direction | 4 | Moving value (-1, 0, 1) |
| Elevator load | 4 | Number of passengers aboard |
| Floor waitingUp | 20 | Boolean per floor |
| Floor waitingDown | 20 | Boolean per floor |
| Floor waitingUpSince | 20 | Steps waiting (0–10,000) |
| Floor waitingDownSince | 20 | Steps waiting (0–10,000) |

Including elevator load in the observation was a deliberate fairness decision: the classical agent has direct access to `elevator.targetFloors` when making decisions, so denying the PPO agent equivalent information would create a structural disadvantage.

#### 2.5 Action Space

```python
MultiDiscrete([3, 3, 3, 3])  # 4 elevators × {idle, up, down}
```

Each step, the agent independently controls the direction of all 4 elevators.

---

### 3. Reward Design

Reward engineering proved to be the most challenging aspect of this project. Several iterations were required before stable learning emerged.

#### 3.1 Iteration History

**Attempt 1 — Raw negative wait time:**
Result: Values reached -6,370,000. Gradient magnitudes were too large for stable learning. `explained_variance ≈ 0`.

**Attempt 2 — Normalized wait penalty:**
```python
reward = -min(avg_wait, 100) / 100  # bounded [-1, 0]
```
Result: More stable, but no positive signal. Agent learned to minimize idling but not to actively serve passengers.

**Attempt 3 — Pickup bonus:**
Result: Agent learned to pick up passengers but not deliver them. Pickup became an end in itself.

**Attempt 4 — Dropoff-weighted reward:**
Result: Partial improvement, but a critical bug was discovered — the reward block was inside the floor iteration loop, causing it to be recalculated 20 times per step, completely corrupting the reward signal.

**Attempt 5 — Shaping reward (final):**
```python
# Per-step directional bonus for each elevator moving toward a target
for elevator in elevators:
    if elevator.targetFloors:
        if moving_toward(elevator, next_target):
            reward += 0.01

# After all floor processing:
reward += dropoffs - min(avg_wait, 100) / 100
```

The shaping reward provides **dense feedback** — the agent receives a small positive signal every step it moves an occupied elevator toward its destination. Without this, rewards were sparse and the agent could go hundreds of steps without meaningful feedback. This is the key insight that enabled stable learning.

#### 3.2 Reward Decomposition

| Component | Range | Purpose |
|---|---|---|
| Shaping bonus | 0 to +0.04 | Dense feedback for correct movement |
| Dropoff bonus | 0 to +N | Reward for completing passenger journeys |
| Wait penalty | -1 to 0 | Penalize accumulated waiting |

---

### 4. Classical Baseline

The classical baseline implements a simplified Destination Dispatching algorithm:

1. **Priority 1:** Elevators with passengers aboard navigate to their next target floor
2. **Priority 2:** Free elevators are assigned to floors with waiting passengers, nearest-first
3. **Assignment tracking:** Each elevator receives at most one assignment per step

**Documented limitations (for transparency):**
- Elevator assignments are not persistent across steps — an elevator can be reassigned before reaching its target
- No look-ahead: the algorithm does not anticipate future demand
- No zone-based load balancing

These limitations are acknowledged deliberately. The baseline represents a reasonable simplified implementation, not a state-of-the-art commercial system. Both agents operate under identical conditions — no advantage is given to either.

---

### 5. PPO Agent

#### 5.1 Architecture

Training used `MlpPolicy` from `stable-baselines3` — a fully connected neural network with two hidden layers of 64 neurons each.

```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0001,
    ent_coef=0.05,
    n_steps=2048,
)
model.learn(total_timesteps=5_000_000)
```

#### 5.2 Key Hyperparameter Decisions

**`ent_coef=0.05`** — A high entropy coefficient was chosen to prevent premature convergence. Early experiments with `ent_coef=0.001` caused the agent to collapse into a near-deterministic policy after ~500k steps, losing exploration capacity before discovering effective strategies. This was diagnosed by monitoring `entropy_loss`, which dropped from -4.0 to near 0 prematurely.

**`learning_rate=0.0001`** — A conservative learning rate for stability.

**`total_timesteps=5,000,000`** — Determined empirically. At 2M steps the agent showed clear improvement but had not plateaued. At 5M steps performance stabilized.

#### 5.3 Training Progression

| Milestone | Steps | ep_rew_mean | explained_variance |
|---|---|---|---|
| Start | 10,240 | -1,220 | 0.39 |
| Early | 20,480 | -645 | 0.43 |
| Midpoint | 2,004,992 | +1,810 | 0.72 |
| Final | 5,001,216 | +1,940 | 0.64 |

The transition from negative to positive `ep_rew_mean` occurred early and held throughout training.

---

### 6. Evaluation Methodology

Both agents were evaluated over **1,000 independent episodes** of exactly 10,000 steps. The `terminated` condition was disabled for evaluation to ensure all episodes run to full length — this eliminates variance from early termination and makes per-step reward comparisons valid across agents.

Reported metrics:
- **Mean Avg Reward:** total episode reward / steps
- **Mean Dropoffs:** total completed passenger journeys per episode
- **Mean Avg Wait:** mean of per-step average wait times across episode

The PPO agent was evaluated with `deterministic=False`. This is appropriate because the agent was trained with a stochastic policy (`ent_coef=0.05`). A stochastic policy also has practical merit in real deployment — slight randomness prevents the system from becoming exploitable by deterministic passenger behavior patterns.

---

### 7. Results

#### 7.1 Quantitative Comparison

|  | Classic Agent | PPO Agent | Improvement |
|---|---|---|---|
| Mean Avg Reward | -0.67 | **+0.14** | +0.81 |
| Mean Dropoffs / Episode | 2,664 | **2,855** | +7.2% |
| Mean Avg Wait (steps) | 600.61 | **93.06** | −84.5% |
| Mean Pickups / Episode | 2,664 | **2,866** | +7.6% |

#### 7.2 Interpretation

The 84.5% reduction in average wait time is the most significant finding. It confirms the PPO agent is not engaging in **reward hacking** — it does not achieve higher reward by shuttling passengers on artificially short trips to accumulate dropoff bonuses. Instead, it genuinely serves passengers faster while also increasing total throughput by 7%.

The near-equal pickup and dropoff counts for the classical agent (2,664 vs 2,664) versus the PPO agent (2,866 vs 2,855) suggest both agents complete nearly all journeys they begin. The PPO agent simply begins more journeys per episode due to more proactive positioning.

---

### 8. Limitations and Future Work

#### 8.1 Current Limitations

**Simplified time model.** The current simulation treats each step as one floor traversal. Real elevators have acceleration curves, door open/close cycles (typically 2–4 seconds per operation), and dwell times for passenger boarding. **This is currently being worked on** — adding realistic elevator kinematics will allow meaningful translation of simulation steps to wall-clock seconds, and will likely change the relative performance of both agents in ways that are worth measuring.

**Boolean waiting model.** Floors track only whether someone is waiting, not how many. This is realistic for the information available to the dispatch system, but means peak-load scenarios with many simultaneous waiting passengers are not fully captured.

**No capacity constraints.** Elevators currently accept unlimited passengers. A realistic capacity of 8–12 persons would change dispatch incentives significantly.

**Single elevator bank.** The simulation assumes one group of 4 elevators serving all floors. Real high-rise buildings typically use express zones and multiple banks.

#### 8.2 Future Work

- Realistic elevator kinematics (acceleration, door cycles, dwell time) — *in progress*
- FastAPI inference server for real-time dispatch
- Next.js frontend for live visualization
- Elevator capacity constraints
- Comparison against additional baselines (SCAN, collective control)
- Multi-bank topologies for high-rise simulation

---

### 9. Conclusion

This project demonstrates that a PPO agent trained from scratch — with no prior knowledge of elevator dispatch heuristics — can significantly outperform a hand-coded Destination Dispatching algorithm in a simulated environment. The key enablers were a well-designed dense reward signal and sufficient training steps to allow the policy to explore and consolidate.

The 84.5% reduction in average passenger wait time, achieved while simultaneously increasing throughput by 7%, suggests the PPO agent has internalized a fundamentally different and superior dispatch strategy: rather than reacting to existing calls, it positions elevators proactively.

---

### References

- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347
- Raffin, A. et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* JMLR
- Brockman, G. et al. (2016). *OpenAI Gym.* arXiv:1606.01540
- Crites, R. H., & Barto, A. G. (1996). *Improving elevator performance using reinforcement learning.* NeurIPS

---

*This project is part of an independent portfolio. All simulation code, training scripts, and evaluation results are fully reproducible from this repository.*