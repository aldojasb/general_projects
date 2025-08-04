# Hands-on Reinforcement Learning Application: PPO for Chemical Process Control

This project illustrates the core concepts of Reinforcement Learning (RL) — such as policy optimization, value estimation, and safe exploration — through the implementation of a **PPO agent** managing an industrial Continuous Stirred Tank Reactor (CSTR). It demonstrates how **RL can lead to significant operational and economic gains** in real-world industrial scenarios.



## **Scenario: Specialty Chemical Manufacturing [Process Industry]**

**Client:** NovaSynauta Labs Inc.

**Facility:** Modular continuous reactor line in British Columbia

**Technology:** CSTR used in exothermic reaction for specialty intermediate synthesis

**Problem:** Maintaining stable product quality and safety margins in an exothermic reaction is challenging. The process is sensitive to feed composition and temperature drift, often requiring manual adjustments by experienced operators.

**Current operations:**

- Feed rates and concentrations are kept within conservative ranges.
- Manual monitoring and correction is frequent, especially during startup or process drift.

**Business implications:**

- Conservative operations lead to **lower throughput and efficiency**.
- Overcorrecting feed composition to prevent overheating can cause **reagent wastage**.

> In continuous chemical operations, **RL enables smart control** that reacts to process dynamics in real time while respecting safety and quality constraints.



## **The Environment: Continuous Stirred Tank Reactor (CSTR)**

We use the **CSTR environment from [PC-Gym](https://maximilianb2.github.io/pc-gym/env/cstr/)**, which models a continuous reactor where two input flows drive a temperature-sensitive, exothermic reaction.

### **CSTR in Industrial Chemistry:**

CSTRs are used across pharmaceuticals, petrochemicals, and specialty chemical production to carry out **steady-state reactions**. The main challenge is maintaining **optimal reactor temperature and concentration** to ensure product quality and safety.



### **Greenhouse Analogy: Climate Control for Fragile Plants**

Imagine a greenhouse filled with delicate tropical plants. These plants grow best within a narrow temperature range — not too hot, not too cold.

- You can't control the sunlight or humidity directly.
- Your only lever is the **cooling system** (like the temperature-controlled vents or fans).
- On hot days, you lower the cooling system temperature to avoid plant stress.
- On cool days, you reduce cooling to keep conditions optimal.

Just like in the CSTR environment, the **RL agent adjusts only the cooling jacket temperature** — similar to how a greenhouse operator tunes the cooling system. The goal is to keep the internal climate **stable and productive**, without wasting energy or risking overheating.



## **The Goal and KPIs**

Primary Goal: **Maximize production efficiency** (reaction conversion) while **maintaining reactor temperature within safe bounds**.

### **Key KPIs for CSTR Optimization:**

| **KPI**                      | **Description**                                     | **Why it matters**                       |
| ---------------------------- | --------------------------------------------------- | ---------------------------------------- |
| **Cumulative Reward**        | Encodes conversion rate vs. penalty for overheating | Measures policy performance              |
| **Temperature Violations**   | Number of steps above safe temperature              | Indicates safety and operational risk    |
| **Conversion Efficiency**    | Output concentration as a function of input         | Reflects reaction success                |
| **Reagent Usage Efficiency** | Product per unit reagent used                       | Tracks economic and sustainability value |
| **Rule-Based Comparison**    | RL vs. static control baseline                      | Highlights learning performance gains    |

### **Comparison to Rule-Based Operations**

The rule-based baseline reflects a conservative logic:

```python
if temperature > 360:
    reduce flow and concentration
else:
    keep steady
```

This avoids runaway reactions, but sacrifices throughput and material efficiency.

Our **RL agent will be evaluated based on whether it can**:

- Anticipate overheating and adjust proactively.
- Use more assertive settings safely when conditions allow.
- Increase output without wasting reagent.
- **Support operators** in managing drift, startup, and transient behaviors.

> Matching or outperforming the baseline proves RL's value as a partner in chemical process optimization.



## **Environment Variables**

At every time step, the environment send a **vector of observations** that describe the internal condition of the reactor. These variables are essential for understanding the chemical reaction dynamics and guiding decisions about temperature control.

| **Observation** | **Unit**   | **Description**                                              |
| --------------- | ---------- | ------------------------------------------------------------ |
| Ca              | mol/L      | Concentration of reactant A. A key input to the reaction. Lower Ca may signal progress or depletion. |
| T               | K (Kelvin) | Reactor temperature. Central to both safety and reaction rate. High T = faster reaction but riskier. |
| Cb              | mol/L      | Concentration of product B. Indicates how much reactant has converted into product. Proxy for yield. |





## **The Levers the Agent Can Pull in this Environment**

The RL agent in the CSTR environment controls a **single continuous action**: the **temperature of the cooling jacket**. This jacket indirectly regulates the reactor temperature and thus the reaction rate and safety.

| **Action**             | **Range**   | **Effect**                                                   |
| ---------------------- | ----------- | ------------------------------------------------------------ |
| **Jacket Temperature** | 290 – 302 K | Adjusts reactor cooling. Lower temperature = more cooling (slows reaction), higher = less cooling (faster reaction, but risk of overheating). |

### **Industrial Interpretation**

- The jacket acts like an external thermostat - controlling the environment surrounding the reactor.
- The agent’s job is to manipulate this temperature smartly to keep the reaction steady and productive.

In real-world operations, operators would adjust cooling rates based on experience or safety protocols. The RL agent learns to do this **continuously and adaptively**, helping maintain stability without overreacting to every fluctuation.



### **Smart Control via RL**

| **Time Step** | **Jacket Temp (K)** | **Reactor Temp (K)** | **Interpretation**                          |
| ------------- | ------------------- | -------------------- | ------------------------------------------- |
| 0             | 295.0               | 340.0                | Stable production. Jacket provides cooling. |
| 5             | 293.5               | 348.0                | Anticipating heat buildup — more cooling.   |
| 10            | 292.0               | 356.5                | Near thermal limit — aggressive cooling.    |
| 15            | 294.5               | 345.0                | Temp stabilized — easing off cooling.       |
| 20            | 296.0               | 350.0                | Maintaining optimal range with fine-tuning. |

### **What the RL Agent Learns:**

- **Proactively lowers jacket temperature** as internal heat accumulates from reaction dynamics.
- **Recovers from overshoot** by adjusting cooling back up gradually.
- Maintains safety and throughput by learning **when and how much to cool** — instead of reacting too late or overcorrecting.

> This illustrates how an RL agent can act as a **predictive control assistant**, adjusting continuously and intelligently even with just **one lever**: temperature regulation.



## **Reward Function: How the Agent Learns**

The reward function is the **core mechanism** that teaches the RL agent how to control the CSTR effectively. Understanding how rewards are calculated is crucial for interpreting agent behavior and optimizing performance.

### **Reward Calculation Formula**

The reward is calculated using a **tracking control** approach that penalizes deviations from target setpoints:

```
Reward = -Σ(r_scale[k] × (state[k] - setpoint[k][t])²)
```

**Where:**
- `state[k]` = Current sensor reading for variable k (e.g., Ca concentration) **after** the action is applied
- `setpoint[k][t]` = Target value for variable k at the current timestep t
- `r_scale[k]` = Reward scaling factor for variable k (makes penalties more significant)
- The negative sign makes it a **penalty** (minimization problem)

### **Variables Explained**

| **Variable** | **Description** | **Example** |
|-------------|----------------|-------------|
| `state[k]` | Current sensor reading after action | Ca = 0.816 mol/L |
| `setpoint[k][t]` | Target value for current timestep | Target Ca = 0.90 mol/L |
| `r_scale[k]` | Reward scaling factor | r_scale['Ca'] = 1000 |
| `t` | Current timestep | t = 1 |

### **Real-World Example**

**Scenario:** Agent applies cooling action, resulting in:
- Current Ca concentration: 0.816 mol/L
- Target Ca concentration: 0.90 mol/L  
- Reward scaling: 1000

**Calculation:**
```
Error = 0.816 - 0.90 = -0.084
Squared Error = (-0.084)² = 0.007056
Reward = -1000 × 0.007056 = -7.06
```

**Interpretation:** The agent receives a **negative reward (-7.06)** because the current concentration (0.816) deviates from the target (0.90). The large magnitude is due to the scaling factor of 1000, which provides stronger learning signals.

### **Why This Reward Design Works**

1. **Encourages Setpoint Tracking:** The agent learns to keep sensor values close to targets
2. **Provides Strong Learning Signal:** The scaling factor makes small deviations meaningful
3. **Supports Real-World Objectives:** Tracks the desired operating trajectory
4. **Enables Safe Operation:** Penalizes deviations that could lead to quality or safety issues

### **Reward Scaling Importance**

The `r_scale` parameter is crucial because:
- **Without scaling:** Small deviations (0.0025) produce tiny rewards (-0.0025)
- **With scaling (1000):** Same deviations produce meaningful rewards (-2.5)
- **Result:** Agent receives stronger feedback for learning optimal control strategies

> The reward function transforms the **tracking control problem** into a reinforcement learning problem that the agent can solve through trial and error.



## **Setpoints: Real-World Operating Targets**

Setpoints represent the **desired operating conditions** that the CSTR should maintain throughout the production process. These are not arbitrary values but carefully designed trajectories based on real-world chemical engineering principles.

### **What Are Setpoints?**

Setpoints are **target values** for key process variables (like concentration) that change over time according to a predefined trajectory. They represent the **optimal operating path** that maximizes yield, quality, and safety.

### **Real-World Significance**

**Who Defines Setpoints:**
- **Process Engineers:** Based on reaction kinetics and thermodynamics
- **Chemical Engineers:** Ensuring safety constraints and optimal conditions  
- **Production Managers:** Balancing quality vs. quantity requirements
- **Quality Control:** Meeting product specifications and regulatory compliance

**Why Setpoints Change Over Time:**
- **Startup Phase:** Safe catalyst activation and gradual temperature increase
- **Production Phase:** Optimal conditions for maximum yield
- **Transition Phase:** Process adjustments for different product requirements
- **Shutdown Phase:** Controlled deactivation and safe cooling

### **Configuration Example**

In your `cstr_environment.yaml`:

```yaml
setpoints:
  ca_profile:
    - value: 0.85         # Startup phase - safe startup conditions
      duration: 3          # Apply for 3 timesteps
    - value: 0.9          # Production phase - maximum yield
      duration: 3          # Apply for 3 timesteps  
    - value: 0.87         # Transition phase - process adjustment
      duration: 4          # Apply for 4 timesteps (fills remaining steps)
```

**Real-World Interpretation:**
- **Timesteps 0-2:** Startup phase with conservative concentration (0.85)
- **Timesteps 3-5:** Production phase with optimal concentration (0.90)
- **Timesteps 6-9:** Transition phase with adjusted concentration (0.87)

### **Real-World Consequences of Setpoint Deviations**

| **Deviation Type** | **Real-World Impact** | **Business Cost** |
|-------------------|----------------------|------------------|
| **Product Quality** | Off-spec material requiring reprocessing | $50K-$200K per batch |
| **Safety Issues** | Runaway reactions, equipment damage | $500K-$2M in damages |
| **Yield Loss** | Reduced product output from same inputs | 10-30% efficiency loss |
| **Environmental** | Waste generation, regulatory violations | Fines + reputation damage |

### **Example: Pharmaceutical Manufacturing**

**Setpoint Trajectory:** `[0.85 → 0.90 → 0.87]`

**Real-World Scenario:**
1. **Startup (0.85):** Safe catalyst activation, gradual temperature increase
2. **Production (0.90):** Optimal conditions for maximum API yield
3. **Transition (0.87):** Process adjustment for different product specifications

**Why This Matters:**
- **Catalyst Safety:** Gradual startup prevents thermal runaway
- **Yield Optimization:** Production phase maximizes conversion
- **Quality Control:** Transition phase ensures product consistency

### **Industrial Control Systems**

In real CSTR operations, multiple control systems work together:

1. **PID Controllers:** Traditional feedback control
2. **Model Predictive Control (MPC):** Advanced control using process models  
3. **Reinforcement Learning:** Adaptive control (like your CSTR environment)

The setpoint trajectory represents the **desired operating path** that these control systems try to follow, ensuring safe, efficient, and profitable operation.

> Setpoints are both **theoretical** (based on chemical engineering principles) and **practical** (representing real-world CSTR control challenges that operators face daily).
