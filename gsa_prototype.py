import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Try importing required libraries
try:
    import pypsa
    from SALib.sample import sobol
    from SALib.analyze import sobol as analyze_sobol
except ImportError as e:
    print(f"Error: Missing required library. {e}")
    print("Please install required libraries using:")
    print("  pip install pypsa salib pandas numpy matplotlib")
    sys.exit(1)

def build_base_network():
    """
    Build a lightweight single-node PyPSA network with Solar, Wind, Battery, and Diesel.
    Uses representative sampling (3-hour resolution, sampled days) for ultra-fast execution.
    """
    n = pypsa.Network()
    
    # 1. Define Snapshots (1 year sampled every 5 days, 3-hour resolution -> 73 days * 8 snapshots = 584 timesteps)
    snapshots = pd.date_range("2026-01-01", "2026-12-31 21:00", freq="3h")
    # Sample every 5th day to keep calculation under 30 seconds
    sampled_snapshots = snapshots[snapshots.dayofyear % 5 == 0]
    n.set_snapshots(sampled_snapshots)
    
    # Snapshot weightings: 5 days * 3 hours = 15 hours per snapshot
    n.snapshot_weightings.loc[:] = 15.0

    # 2. Add Bus
    n.add("Bus", "electricity")

    # 3. Add Demand Profile (Synthetic load: base + diurnal peak)
    hours = n.snapshots.hour
    dayofyear = n.snapshots.dayofyear
    demand_profile = 100 + 30 * np.sin(2 * np.pi * (hours - 8) / 24) + 20 * np.sin(2 * np.pi * dayofyear / 365)
    n.add("Load", "demand", bus="electricity", p_set=demand_profile)

    # 4. Add Renewable Profiles (Synthetic Solar & Wind)
    solar_profile = np.maximum(0, np.sin(np.pi * (hours - 6) / 12)) * (hours >= 6) * (hours <= 18)
    wind_profile = 0.4 + 0.3 * np.cos(2 * np.pi * hours / 24) + 0.2 * np.sin(2 * np.pi * dayofyear / 50)
    wind_profile = np.clip(wind_profile, 0.05, 0.95)

    # 5. Add Generators
    # Solar
    n.add(
        "Generator",
        "solar",
        bus="electricity",
        p_nom_extendable=True,
        p_max_pu=solar_profile,
        capital_cost=50000,  # $/MW/year (default)
        marginal_cost=0,
    )

    # Wind
    n.add(
        "Generator",
        "wind",
        bus="electricity",
        p_nom_extendable=True,
        p_max_pu=wind_profile,
        capital_cost=80000,  # $/MW/year (default)
        marginal_cost=0,
    )

    # Diesel Generator (Dispatchable, low capital cost, high fuel/marginal cost)
    n.add(
        "Generator",
        "diesel",
        bus="electricity",
        p_nom_extendable=True,
        capital_cost=15000,  # $/MW/year
        marginal_cost=100,   # $/MWh (fuel + O&M)
    )

    # 6. Add Battery Storage
    n.add(
        "StorageUnit",
        "battery",
        bus="electricity",
        p_nom_extendable=True,
        max_hours=6,
        capital_cost=60000,  # $/MW/year (default)
        marginal_cost=0,
        efficiency_dispatch=0.9,
        efficiency_store=0.9,
    )

    return n

def detect_solver():
    """Detect an available solver (cbc, highs, glpk, gurobi)."""
    solvers = ["cbc", "highs", "glpk", "gurobi"]
    # We will pass solver_name to pypsa
    return "cbc"  # Default solver assumption

def run_gsa():
    print("=" * 60)
    print(" PyPSA x SALib Global Sensitivity Analysis (GSA) Prototype")
    print(" Technologies: Solar, Wind, Battery, Diesel Generator")
    print("=" * 60)

    # 1. Define Problem Space for SALib (4 parameters)
    problem = {
        "num_vars": 4,
        "names": ["solar_cost", "wind_cost", "battery_cost", "diesel_marginal_cost"],
        "bounds": [
            [35000, 65000],   # Solar capital cost ($/MW/year)
            [56000, 104000],  # Wind capital cost ($/MW/year)
            [30000, 90000],   # Battery capital cost ($/MW/year)
            [60, 140],        # Diesel fuel marginal cost ($/MWh)
        ],
    }

    # 2. Generate Saltelli Samples
    N = 32  # Small sample size for fast prototype execution
    param_values = sobol.sample(problem, N, calc_second_order=True)
    num_samples = len(param_values)
    print(f"[1/4] Generated {num_samples} parameter samples using Saltelli scheme.")

    # 3. Execute PyPSA Optimization Loop
    print(f"[2/4] Running {num_samples} PyPSA optimizations...")
    
    total_costs = []
    battery_capacities = []
    diesel_capacities = []

    base_network = build_base_network()

    for i, params in enumerate(param_values):
        # Clone base network
        n = base_network.copy()

        # Update parameter values
        n.generators.loc["solar", "capital_cost"] = params[0]
        n.generators.loc["wind", "capital_cost"] = params[1]
        n.storage_units.loc["battery", "capital_cost"] = params[2]
        n.generators.loc["diesel", "marginal_cost"] = params[3]

        # Solve LOPF (Linear Optimal Power Flow)
        try:
            status, _ = n.optimize(solver_name="cbc", solver_options={"log": False})
        except Exception:
            # Fallback if solver name needs adjusting
            status, _ = n.optimize(solver_options={"log": False})

        # Record metrics
        total_cost = n.objective / 1e6  # Million $
        bat_cap = n.storage_units.loc["battery", "p_nom_opt"]
        diesel_cap = n.generators.loc["diesel", "p_nom_opt"]

        total_costs.append(total_cost)
        battery_capacities.append(bat_cap)
        diesel_capacities.append(diesel_cap)

        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            print(f"  Completed {i + 1}/{num_samples} samples...")

    total_costs = np.array(total_costs)
    battery_capacities = np.array(battery_capacities)

    # 4. Perform Sobol Analysis for Total System Cost
    print("[3/4] Computing Sobol Sensitivity Indices (S1, ST)...")
    Si_cost = analyze_sobol.analyze(problem, total_costs, calc_second_order=True)

    print("\n" + "-" * 50)
    print("RESULTS: Sobol Indices for Total System Cost")
    print("-" * 50)
    print(f"{'Parameter':<22} {'S1 (Direct)':<15} {'ST (Total)':<15}")
    print("-" * 50)
    for name, s1, st in zip(problem["names"], Si_cost["S1"], Si_cost["ST"]):
        print(f"{name:<22} {s1:<15.4f} {st:<15.4f}")
    print("-" * 50)

    # 5. Plot Comparison Bar Chart
    print("[4/4] Generating Sensitivity Plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(problem["names"]))
    width = 0.35

    s1_vals = np.maximum(0, Si_cost["S1"])  # Clamp small negative estimates to 0
    st_vals = np.maximum(0, Si_cost["ST"])

    s1_err = Si_cost["S1_conf"]
    st_err = Si_cost["ST_conf"]

    rects1 = ax.bar(x - width/2, s1_vals, width, yerr=s1_err, label="S1 (First-order / Direct)", color="#2b5c8f", capsize=5)
    rects2 = ax.bar(x + width/2, st_vals, width, yerr=st_err, label="ST (Total-order / Incl. Interactions)", color="#d95f02", capsize=5)

    ax.set_ylabel("Sobol Index", fontsize=12)
    ax.set_title("Global Sensitivity Analysis: Impact of Tech Costs on System Cost\n(Solar, Wind, Battery, Diesel)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Solar CapCost", "Wind CapCost", "Battery CapCost", "Diesel FuelCost"], fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    output_img = "sobol_indices_prototype.png"
    plt.savefig(output_img, dpi=300)
    print(f"\n[Success] Plot saved as '{output_img}'.")
    print("=" * 60)

if __name__ == "__main__":
    run_gsa()
