# ==============================================================================
# Sequential Optimization via RHS Updates: 24-Hour Battery Storage Benchmark
# ------------------------------------------------------------------------------
# Academic Inspiration: Warren Powell's Sequential Decision Analytics (SDA)
# 
# Demonstrates the performance gap between:
#   1. Naive Rebuild Approach (Anti-Pattern): Re-allocating model memory and re-building 
#      constraints at every sequential time step.
#   2. Persistent Model RHS Updates (Best Practice): Building the constraint 
#      matrix ONCE and mutating Right-Hand Side (RHS) pointers in-place.
# ==============================================================================

using JuMP
using HiGHS
using Printf
using Dates

# ------------------------------------------------------------------------------
# System Parameters & Forecast Data
# ------------------------------------------------------------------------------
struct BatterySystem
    capacity_mwh::Float64   # Storage capacity E_max (MWh)
    p_max_mw::Float64       # Max charge/discharge rate P_max (MW)
    eta_charge::Float64     # Charging efficiency
    eta_discharge::Float64  # Discharging efficiency
    soc_min::Float64        # Min state of charge (fraction)
    soc_max::Float64        # Max state of charge (fraction)
end

function default_battery()
    return BatterySystem(100.0, 25.0, 0.95, 0.95, 0.10, 0.90)
end

# Generate realistic 24-hour electricity prices ($/MWh) with two daily peaks, repeated over T hours
function generate_price_profile(T::Int=8760)
    hours = 1:T
    day_hour = (hours .- 1) .% 24 .+ 1
    base_price = 35.0
    morning_peak = 45.0 .* exp.(-0.5 .* ((day_hour .- 8) ./ 2.0).^2)
    evening_peak = 75.0 .* exp.(-0.5 .* ((day_hour .- 19) ./ 2.5).^2)
    solar_dip = -15.0 .* exp.(-0.5 .* ((day_hour .- 13) ./ 2.0).^2)
    return base_price .+ morning_peak .+ evening_peak .+ solar_dip
end

# ------------------------------------------------------------------------------
# Approach 1: Naive Rebuild (Anti-Pattern)
# ------------------------------------------------------------------------------
function run_naive_benchmark(battery::BatterySystem, price_profile::Vector{Float64}, horizon::Int=24)
    T = length(price_profile)
    current_soc = battery.capacity_mwh * 0.50 # Start at 50% SOC
    
    total_setup_time_ms = 0.0
    total_solve_time_ms = 0.0
    total_allocations_bytes = 0
    
    dispatch_history = Float64[]
    soc_history = [current_soc]
    total_profit = 0.0
    
    for t in 1:T
        # Determine rolling horizon price window
        window_len = min(horizon, T - t + 1)
        prices = price_profile[t:(t + window_len - 1)]
        
        # Benchmark Model Construction
        setup_stats = @timed begin
            model = Model(HiGHS.Optimizer)
            set_silent(model)
            
            # Variables
            @variable(model, 0 <= p_c[1:window_len] <= battery.p_max_mw)
            @variable(model, 0 <= p_d[1:window_len] <= battery.p_max_mw)
            @variable(model, battery.capacity_mwh * battery.soc_min <= soc[1:window_len] <= battery.capacity_mwh * battery.soc_max)
            
            # Initial condition constraint (Re-created at every step!)
            @constraint(model, init_soc, soc[1] == current_soc + battery.eta_charge * p_c[1] - (1.0 / battery.eta_discharge) * p_d[1])
            
            # Dynamics over horizon
            for tau in 2:window_len
                @constraint(model, soc[tau] == soc[tau-1] + battery.eta_charge * p_c[tau] - (1.0 / battery.eta_discharge) * p_d[tau])
            end
            
            # Objective: Maximize Arbitrage Revenue
            @objective(model, Max, sum(prices[tau] * (p_d[tau] - p_c[tau]) for tau in 1:window_len))
            model
        end
        
        total_setup_time_ms += setup_stats.time * 1000.0
        total_allocations_bytes += setup_stats.bytes
        
        # Benchmark Solver Solve
        model_inst = setup_stats.value
        solve_stats = @timed optimize!(model_inst)
        
        total_solve_time_ms += solve_time(model_inst) * 1000.0
        total_allocations_bytes += solve_stats.bytes
        
        # Extract immediate decision (First step of look-ahead)
        p_c_val = value(model_inst[:p_c][1])
        p_d_val = value(model_inst[:p_d][1])
        net_power = p_d_val - p_c_val
        
        # Update physical system state for step t+1
        current_soc += battery.eta_charge * p_c_val - (1.0 / battery.eta_discharge) * p_d_val
        push!(soc_history, current_soc)
        push!(dispatch_history, net_power)
        total_profit += prices[1] * net_power
    end
    
    return (
        setup_time_ms = total_setup_time_ms,
        solve_time_ms = total_solve_time_ms,
        total_time_ms = total_setup_time_ms + total_solve_time_ms,
        allocations_mb = total_allocations_bytes / (1024 * 1024),
        total_profit = total_profit,
        soc_history = soc_history
    )
end

# ------------------------------------------------------------------------------
# Approach 2: Persistent Model with In-Place RHS Updates (Best Practice)
# ------------------------------------------------------------------------------
function run_persistent_rhs_benchmark(battery::BatterySystem, price_profile::Vector{Float64}, horizon::Int=24)
    T = length(price_profile)
    current_soc = battery.capacity_mwh * 0.50
    
    total_setup_time_ms = 0.0
    total_update_time_ms = 0.0
    total_solve_time_ms = 0.0
    total_allocations_bytes = 0
    
    dispatch_history = Float64[]
    soc_history = [current_soc]
    total_profit = 0.0
    
    # 1. BUILD MODEL EXACTLY ONCE OUTSIDE THE LOOP
    init_stats = @timed begin
        model = Model(HiGHS.Optimizer)
        set_silent(model)
        
        @variable(model, 0 <= p_c[1:horizon] <= battery.p_max_mw)
        @variable(model, 0 <= p_d[1:horizon] <= battery.p_max_mw)
        @variable(model, battery.capacity_mwh * battery.soc_min <= soc[1:horizon] <= battery.capacity_mwh * battery.soc_max)
        
        # Storage dynamics
        for tau in 2:horizon
            @constraint(model, soc[tau] - soc[tau-1] - battery.eta_charge * p_c[tau] + (1.0 / battery.eta_discharge) * p_d[tau] == 0.0)
        end
        
        # Initial SOC constraint reference (RHS will be updated in-place!)
        # Form: soc[1] - eta_c * p_c[1] + (1/eta_d) * p_d[1] == initial_soc_val
        @constraint(model, init_soc_con, soc[1] - battery.eta_charge * p_c[1] + (1.0 / battery.eta_discharge) * p_d[1] == current_soc)
        
        (model=model, init_soc_con=init_soc_con, p_c=p_c, p_d=p_d)
    end
    
    total_setup_time_ms = init_stats.time * 1000.0
    total_allocations_bytes += init_stats.bytes
    
    model = init_stats.value.model
    init_soc_con = init_stats.value.init_soc_con
    p_c = init_stats.value.p_c
    p_d = init_stats.value.p_d
    
    # 2. SEQUENTIAL DECISION LOOP WITH IN-PLACE RHS & OBJECTIVE UPDATES
    for t in 1:T
        window_len = min(horizon, T - t + 1)
        prices = price_profile[t:(t + window_len - 1)]
        
        # Update RHS and Objective Coefficients in-place
        update_stats = @timed begin
            # Mutate the initial condition RHS in solver memory pointer
            set_normalized_rhs(init_soc_con, current_soc)
            
            # Update objective coefficients for remaining window
            for tau in 1:horizon
                if tau <= window_len
                    set_objective_coefficient(model, p_d[tau], prices[tau])
                    set_objective_coefficient(model, p_c[tau], -prices[tau])
                else
                    # Zero out beyond window horizon
                    set_objective_coefficient(model, p_d[tau], 0.0)
                    set_objective_coefficient(model, p_c[tau], 0.0)
                end
            end
            set_objective_sense(model, MOI.MAX_SENSE)
        end
        
        total_update_time_ms += update_stats.time * 1000.0
        total_allocations_bytes += update_stats.bytes
        
        # Solve with Warm Start (retaining basis)
        solve_stats = @timed optimize!(model)
        
        total_solve_time_ms += solve_time(model) * 1000.0
        total_allocations_bytes += solve_stats.bytes
        
        # Extract optimal immediate action
        p_c_val = value(p_c[1])
        p_d_val = value(p_d[1])
        net_power = p_d_val - p_c_val
        
        # Physics update for next period
        current_soc += battery.eta_charge * p_c_val - (1.0 / battery.eta_discharge) * p_d_val
        push!(soc_history, current_soc)
        push!(dispatch_history, net_power)
        total_profit += prices[1] * net_power
    end
    
    return (
        setup_time_ms = total_setup_time_ms,
        update_time_ms = total_update_time_ms,
        solve_time_ms = total_solve_time_ms,
        total_time_ms = total_setup_time_ms + total_update_time_ms + total_solve_time_ms,
        allocations_mb = total_allocations_bytes / (1024 * 1024),
        total_profit = total_profit,
        soc_history = soc_history
    )
end

# ------------------------------------------------------------------------------
# Benchmark Runner & Formatted Reporting
# ------------------------------------------------------------------------------
function main()
    println("================================================================================")
    println(" SEQUENTIAL OPTIMIZATION BENCHMARK: NAIVE REBUILD vs PERSISTENT RHS UPDATE")
    println(" Language: Julia $(VERSION) | Solver: HiGHS (via JuMP.jl)")
    println(" Problem: 8760-Hour (1-Year) Battery Storage Arbitrage (Rolling Horizon Look-Ahead)")
    println("================================================================================")
    println()

    battery = default_battery()
    prices = generate_price_profile(8760)

    # Warmup runs to eliminate Julia JIT compilation overhead
    print("Warming up JIT compiler...")
    run_naive_benchmark(battery, prices[1:6], 6)
    run_persistent_rhs_benchmark(battery, prices[1:6], 6)
    println(" Done.")
    println()

    # Benchmark Execution
    println("Running Approach 1: Naive Rebuild Approach (8760 Steps)...")
    res_naive = run_naive_benchmark(battery, prices, 24)
    
    println("Running Approach 2: Persistent RHS Update Approach (8760 Steps)...")
    res_persistent = run_persistent_rhs_benchmark(battery, prices, 24)

    # Compute Acceleration & Savings
    build_speedup = res_naive.setup_time_ms / res_persistent.update_time_ms
    solve_speedup = res_naive.solve_time_ms / res_persistent.solve_time_ms
    total_speedup = res_naive.total_time_ms / res_persistent.total_time_ms
    mem_reduction = ((res_naive.allocations_mb - res_persistent.allocations_mb) / res_naive.allocations_mb) * 100.0

    # Output Benchmark Summary Table
    println()
    println("┌─────────────────────────────────────────┬──────────────────┬──────────────────┬──────────────┐")
    println("│ Performance Metric                      │ Naive Rebuild    │ Persistent RHS   │ Speedup/Gain │")
    println("├─────────────────────────────────────────┼──────────────────┼──────────────────┼──────────────┤")
    @printf("│ %-39s │ %16.3f │ %16.3f │ %11.1fx │\n", "Structural Build / Setup Time (ms)", res_naive.setup_time_ms, res_persistent.setup_time_ms, res_naive.setup_time_ms / res_persistent.setup_time_ms)
    @printf("│ %-39s │ %16s │ %16.3f │ %12s │\n", "Step RHS & Obj Update Time (ms)", "N/A (Rebuilt)", res_persistent.update_time_ms, "Near-Zero")
    @printf("│ %-39s │ %16.3f │ %16.3f │ %11.1fx │\n", "Pure Solver Solve Time (ms)", res_naive.solve_time_ms, res_persistent.solve_time_ms, solve_speedup)
    @printf("│ %-39s │ %16.3f │ %16.3f │ %11.1fx │\n", "Total Execution Time (ms)", res_naive.total_time_ms, res_persistent.total_time_ms, total_speedup)
    @printf("│ %-39s │ %16.2f │ %16.2f │ %11.1f%% │\n", "Total Memory Allocated (MB)", res_naive.allocations_mb, res_persistent.allocations_mb, mem_reduction)
    println("└─────────────────────────────────────────┴──────────────────┴──────────────────┴──────────────┘")
    println()

    # Numerical Validation Check
    profit_diff = abs(res_naive.total_profit - res_persistent.total_profit)
    soc_diff = maximum(abs.(res_naive.soc_history .- res_persistent.soc_history))
    
    println("┌──────────────────────────────────────────────────────────────────────────────┐")
    println("│ NUMERICAL VALIDATION                                                         │")
    println("├──────────────────────────────────────────────────────────────────────────────┤")
    @printf("│ %-28s: \$%14.2f %-30s │\n", "Naive Total Revenue", res_naive.total_profit, "")
    @printf("│ %-28s: \$%14.2f %-30s │\n", "Persistent Total Revenue", res_persistent.total_profit, "")
    @printf("│ %-28s:  %14.2e %-30s │\n", "Max SOC Trajectory Diff", soc_diff, "MWh")
    if profit_diff < 1e-4 && soc_diff < 1e-4
        @printf("│ %-28s: %-46s │\n", "Status", "[PASSED] Both methods yield identical solutions")
    else
        @printf("│ %-28s: %-46s │\n", "Status", "[FAILED] Discrepancy detected")
    end
    println("└──────────────────────────────────────────────────────────────────────────────┘")
    println()
end

main()
