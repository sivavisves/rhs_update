# Sequential Optimization via In-Place RHS Updates ⚡

[![Julia](https://img.shields.io/badge/Julia-1.12+-9558B2?style=flat&logo=julia)](https://julialang.org)
[![Solver](https://img.shields.io/badge/Solver-HiGHS-blue?style=flat)](https://highs.dev)
[![Framework](https://img.shields.io/badge/JuMP.jl-v1.30+-0570b0?style=flat)](https://jump.dev)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python)](https://python.org)

> Computational benchmark and architectural guide demonstrating **microsecond-level acceleration** in sequential optimization problems (Direct Look-Ahead models, Model Predictive Control, and Rolling Horizon dispatch) by replacing **Naive Model Rebuilding** with **Persistent Models and In-Place Right-Hand Side (RHS) updates**.

---

## 💡 Executive Summary & Academic Inspiration

In sequential decision-making under uncertainty — formalized by **Prof. Warren Powell** in *Sequential Decision Analytics* — decisions $x_t$ are solved sequentially at time steps $t \in \{1, 2, \dots, T\}$.

At each time step $t$, the system receives updated state information $S_t$ (e.g., initial battery state of charge $SOC_t$ or updated market price forecasts). 

### The Performance Dilemma:
1. **Naive Rebuild Approach (Anti-Pattern)**: Re-allocates variable objects, constraint expressions, and constraint matrices from scratch at every step $t$. This incurs a massive **"Object Tax"** (memory reallocation overhead) and forces solvers into **Cold Starts**.
2. **Persistent Model RHS Update Approach (Best Practice)**: Instantiates the optimization model structure and constraint matrix $A$ **exactly once outside the loop**. At each time step $t$, targeted functions mutate constraint Right-Hand Side pointers ($b$) in-place, preserving solver internal memory and enabling **HiGHS Simplex Basis Warm Starts**.

---

## 📊 Benchmark Results

Running the 24-hour battery storage arbitrage benchmark comparing 24 sequential decision solves:

### Julia Benchmark Results (`JuMP.jl` + `HiGHS.jl`)

| Performance Metric | Naive Rebuild Approach | Persistent RHS Approach | Speedup / Gain |
| :--- | :---: | :---: | :---: |
| **Structural Setup / Build Time (ms)** | `4.804 ms` | `0.243 ms` | **19.8x Acceleration** |
| **Step RHS Update Time (ms)** | *N/A (Full Rebuild)* | `0.488 ms` | **Microsecond Scale** |
| **Pure Solver Solve Time (ms)** | `8.448 ms` | `6.048 ms` | **1.4x Faster (Warm Start)** |
| **Total Pipeline Time (ms)** | `13.252 ms` | `6.779 ms` | **2.0x Total Pipeline Speedup** |
| **Total Memory Allocated (MB)** | `3.49 MB` | `0.30 MB` | **91.5% Memory Cut** |

### Numerical Validation
- **Naive Total Revenue**: `$9,696.47`
- **Persistent RHS Revenue**: `$9,696.47`
- **Max Trajectory Difference**: `3.20e-14 MWh` *(Exact numerical equivalence)*

---

## 🏛️ System Architecture & Mechanics

### 1. Sequential Pipeline Comparison
![Pipeline Architecture](assets/diagram_architecture.svg)

### 2. Matrix Mechanics & Warm Starts
![Matrix Warm Start Mechanics](assets/diagram_warmstart.svg)

Mathematically, in the linear program:
$$\min_{x} \, c^T x \quad \text{subject to} \quad A x = b, \quad l \le x \le u$$

- **Constraint Matrix $A$**: Encodes physical dynamics (battery efficiency $\eta_c, \eta_d$, storage energy balance). Matrix $A$ is **structurally invariant** over time.
- **RHS Vector $b$**: Encodes initial state boundary conditions ($SOC_0 = SOC_t$). 
- Modifying vector $b$ shifts the feasible region hyperplanes without changing their normal vectors (slopes).
- **HiGHS Basis Warm Start**: Because $A$ is untouched, HiGHS reuses the optimal basis matrix $B^{-1}$ from step $t-1$, reducing step solves to 1–2 dual simplex pivots.

---

## 📁 Repository Structure

```text
rhs_update/
├── benchmark.jl            # Primary Julia benchmark (JuMP + HiGHS)
├── benchmark.py            # Python benchmark illustrating object tax vs in-place array updates
├── README.md               # Repository documentation
└── assets/
    ├── linkedin_carousel.md # Concept layout for 6-slide LinkedIn presentation
    ├── diagram_architecture.svg # Architectural pipeline SVG diagram
    └── diagram_warmstart.svg   # Linear programming matrix math SVG diagram
```

---

## 🚀 Quickstart & Usage

### Running the Julia Benchmark
Ensure Julia 1.9+ is installed:

```bash
julia benchmark.jl
```

### Running the Python Benchmark
Requires Python 3.8+:

```bash
python3 benchmark.py
```

---

## 📖 Code Snippet (JuMP.jl Best Practice)

```julia
using JuMP, HiGHS

# 1. Instantiate persistent model ONCE outside the sequential loop
model = Model(HiGHS.Optimizer)
set_silent(model)

@variable(model, 0 <= p_c[1:24] <= 25.0)
@variable(model, 0 <= p_d[1:24] <= 25.0)
@variable(model, 10.0 <= soc[1:24] <= 90.0)

# Store reference to initial condition constraint
@constraint(model, init_soc_con, soc[1] - 0.95*p_c[1] + (1/0.95)*p_d[1] == initial_soc)

# 2. Sequential decision loop (Microsecond updates)
for t in 1:24
    # Mutate RHS pointer in-place
    set_normalized_rhs(init_soc_con, current_soc)
    
    # Solve with HiGHS warm start
    optimize!(model)
    
    # Extract decision and update physical state
    current_soc += 0.95 * value(p_c[1]) - (1/0.95) * value(p_d[1])
end
```

---

## 📚 References & Further Reading
1. Powell, Warren B. *Sequential Decision Analytics and Modeling: Modeling with Technology*. Wiley, 2022.
2. JuMP Documentation: [Modifying numeric coefficients in JuMP](https://jump.dev/JuMP.jl/stable/manual/constraints/#Modifying-a-variable-coefficient)
3. HiGHS Solver: [Open-source high-performance LP/MIP solver](https://highs.dev)
