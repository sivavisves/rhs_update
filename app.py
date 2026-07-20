import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from benchmark import BatterySystem, generate_price_profile, run_naive_python_benchmark, run_persistent_python_benchmark

st.set_page_config(page_title="Optimization Benchmark", layout="wide")

st.title("Sequential Optimization: Naive Rebuild vs RHS Update")
st.markdown("""
This app evaluates the computational performance of two different approaches for solving **Sequential Optimization** problems, focusing on a 24-Hour Battery Storage Arbitrage scenario.
""")

with st.expander("ℹ️ About the Optimization Problem & Architecture", expanded=False):
    st.markdown(r"""
    ### Problem Overview
    In sequential decision-making (like Direct Look-Ahead models, MPC, or Rolling Horizon dispatch), decisions are solved sequentially over time. At each step $t$, the system receives updated state information (e.g., current battery State of Charge $SOC_t$ or updated market prices).
    
    While dynamic programming suffers from the "Curse of Dimensionality," applied sequential optimization faces the **Curse of Model Build Time**. Before a solver can optimize, Algebraic Modeling Languages (AMLs) like Pyomo or JuMP must translate the algebraic equations into a flat, numerical matrix. For massive sequential problems, this matrix generation step often takes longer than the optimization itself.

    This benchmark models a **Battery Storage Arbitrage** problem where the system charges during low prices and discharges during high prices to maximize profit.

    ### Mathematical Formulation
    The core linear program solved at each step is:
    $$
    \min_{x} \, c^T x \quad \text{subject to} \quad A x = b, \quad l \le x \le u
    $$

    - **Constraint Matrix ($A$)**: Encodes physical dynamics like battery efficiency ($\eta_c, \eta_d$) and storage energy balance. Matrix $A$ is **structurally invariant** over time.
    - **RHS Vector ($b$)**: Encodes initial state boundary conditions ($SOC_0 = SOC_t$), updated forecasts, or reserve requirements. Modifying $b$ shifts the feasible region hyperplanes without changing their normal vectors (slopes).

    ### The Performance Dilemma
    1. **Naive Rebuild Approach (Anti-Pattern)**: Re-allocates variable objects, constraint expressions, and constraint matrices from scratch at every step. This triggers massive **Symbolic Overhead** and garbage collection pauses as the AML builds out millions of object nodes, mapping them to the solver's C-level API over and over. You pay the matrix generation penalty repeatedly and force the solver into a **Cold Start**.
    2. **Persistent Model RHS Update (Best Practice)**: The constraint matrix $A$ and variable bounds are compiled and loaded into the solver's memory exactly once outside the loop. At each time step, targeted functions mutate the Right-Hand Side (RHS) pointers in-place using low-level API hooks, bypassing AML object creation entirely. 
    3. **Solver Warm-Starting**: Because the matrix $A$ is untouched, solvers (like HiGHS or Gurobi) reuse the optimal basis matrix $B^{-1}$ or interior-point state from the previous step. This reduces subsequent solves to just a few dual simplex pivots, trading a massive memory bottleneck for streamlined solve times!
    """)

with st.expander("ℹ️ The Battery Optimization Problem", expanded=False):
    st.markdown(r"""
    The problem evaluates a daily operation of a grid-scale battery storage system facing wholesale electricity market prices.

    **Objective Function:**
    Maximize the profit from energy arbitrage over the horizon $T$:
    $$
    \max \sum_{t=1}^T \lambda_t \cdot (p_t^{discharge} - p_t^{charge})
    $$
    where $\lambda_t$ is the market price at time $t$, and $p_t$ represents the power discharged/charged.

    **Constraints:**
    1. **State of Charge (SOC) Update:**
       $$ SOC_{t} = SOC_{t-1} + \eta_c p_t^{charge} - \frac{1}{\eta_d} p_t^{discharge} \quad \forall t \in \{1, \dots, T\} $$
    2. **Energy Limits:**
       $$ SOC_{min} \le SOC_t \le SOC_{max} \quad \forall t \in \{1, \dots, T\} $$
    3. **Power Limits:**
       $$ 0 \le p_t^{charge} \le P_{max}^{charge} \quad \forall t \in \{1, \dots, T\} $$
       $$ 0 \le p_t^{discharge} \le P_{max}^{discharge} \quad \forall t \in \{1, \dots, T\} $$

    **Sequential Update & Future Uncertainty:**
    In a real-world rolling horizon dispatch, the future prices $\lambda_t$ are uncertain and updated periodically based on new forecasts. 
    1. At hour $h$, we solve the optimization problem for the next $T$ hours (the look-ahead horizon) using the latest price forecast and the *current* physical battery $SOC_0$.
    2. We implement the dispatch decision for only the first step ($h$).
    3. The horizon rolls forward to $h+1$. The $SOC_0$ is updated to reflect the physical state, new price forecasts $\lambda_t$ are received, and the optimization problem is solved again.
    
    This sequential updating means the mathematical model structure (the matrix) remains constant, but the initial state and parameters (the RHS vector and objective coefficients) change at every step.
    """)

st.sidebar.header("Simulation Parameters")
# Users can increase the number of timesteps into the future (horizon / T)
timesteps = st.sidebar.slider("Number of Timesteps (T)", min_value=12, max_value=720, value=24, step=12)

# Run benchmark when button is clicked
if st.sidebar.button("Run Benchmark"):
    with st.spinner(f"Running benchmarks for {timesteps} timesteps..."):
        battery = BatterySystem()
        prices = generate_price_profile(timesteps)
        
        # Warmup to avoid JIT/initialization overhead affecting results
        run_naive_python_benchmark(battery, generate_price_profile(12), 12)
        run_persistent_python_benchmark(battery, generate_price_profile(12), 12)
        
        # Actual benchmark
        res_naive = run_naive_python_benchmark(battery, prices, timesteps)
        res_persistent = run_persistent_python_benchmark(battery, prices, timesteps)
        
        # Prepare data for plotting
        naive_build_time = res_naive["setup_time_ms"]
        naive_solve_time = res_naive["solve_time_ms"]
        
        # For persistent approach, build time is setup (one-off) + update (in loop)
        rhs_build_time = res_persistent["setup_time_ms"] + res_persistent["update_time_ms"]
        rhs_solve_time = res_persistent["solve_time_ms"]
        
        data = {
            "Time Component": ["Build Time", "Solve Time", "Build Time", "Solve Time"],
            "Approach": ["Naive Rebuild", "Naive Rebuild", "RHS Update", "RHS Update"],
            "Time (ms)": [naive_build_time, naive_solve_time, rhs_build_time, rhs_solve_time]
        }
        
        df = pd.DataFrame(data)
        
        st.subheader("Performance Comparison")
        
        # Create a grouped bar chart
        fig = px.bar(
            df, 
            x="Time Component", 
            y="Time (ms)", 
            color="Approach", 
            barmode="group",
            title=f"Benchmark Results for {timesteps} Timesteps",
            text_auto='.2f',
            color_discrete_sequence=["#EF553B", "#00CC96"]
        )
        
        fig.update_layout(
            xaxis_title="Pipeline Stage",
            yaxis_title="Execution Time (ms)",
            legend_title="Optimization Approach"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display raw data
        st.subheader("Detailed Metrics (ms)")
        metrics_df = pd.DataFrame({
            "Metric": ["Build/Setup Time", "Solve Time", "Total Time"],
            "Naive Rebuild": [naive_build_time, naive_solve_time, res_naive["total_time_ms"]],
            "RHS Update": [rhs_build_time, rhs_solve_time, res_persistent["total_time_ms"]]
        })
        
        metrics_df["Speedup (Naive/RHS)"] = metrics_df["Naive Rebuild"] / metrics_df["RHS Update"]
        
        st.dataframe(metrics_df.style.format({
            "Naive Rebuild": "{:.2f}",
            "RHS Update": "{:.2f}",
            "Speedup (Naive/RHS)": "{:.2f}x"
        }))
else:
    st.info("Click 'Run Benchmark' in the sidebar to evaluate the performance.")
