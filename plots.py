import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from security_sim import (
    FIXED_LANE_OPTIONS,
    DeadlineAwarePolicy,
    FlightBankPolicy,
    MAX_LANES,
    FixedPolicy,
    HysteresisPolicy,
    LookaheadPolicy,
    QueueThresholdPolicy,
    TargetClearancePolicy,
    generate_day,
    pareto_frontier,
    simulate,
)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.8,
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})

BEST_COLOR = "#1f7a5c"
WORST_COLOR = "#b83b3b"
RESULT_COLOR = "#8c9399"
OTHER_COLOR = "#a7abb0"

scenario = generate_day(7)
policies = [FixedPolicy(lanes) for lanes in FIXED_LANE_OPTIONS]
policies += [QueueThresholdPolicy(), LookaheadPolicy(), HysteresisPolicy(), TargetClearancePolicy(),
             FlightBankPolicy(), DeadlineAwarePolicy()]
results = [simulate(scenario, p) for p in policies]

best = min(results, key=lambda r: r.objective)
worst = max(results, key=lambda r: r.objective)
frontier = sorted(pareto_frontier(results), key=lambda r: r.lane_hours)

print("Best :", best.policy, f"obj={best.objective:.1f} late={best.late_rate:.2%} lane-hours={best.lane_hours:.1f}")
print("Worst:", worst.policy, f"obj={worst.objective:.1f} late={worst.late_rate:.2%} lane-hours={worst.lane_hours:.1f}")

fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.3], hspace=0.5, wspace=0.28)

# --- Panel 1: queue length across the day ---
ax1 = fig.add_subplot(gs[0, :])
for flight in scenario.flights:
    ax1.axvline(flight.departure, color="#eeeeee", lw=0.5, zorder=0)
ax1.plot(worst.history["time"], worst.history["queue"], color=WORST_COLOR, lw=1.7,
          label=f"{worst.policy} (worst)")
ax1.plot(best.history["time"], best.history["queue"], color=BEST_COLOR, lw=1.7,
          label=f"{best.policy} (best)")
for r in results:
    if r is not best and r is not worst:
        ax1.plot(r.history["time"], r.history["queue"], color=RESULT_COLOR, lw=1.7, alpha=0.25)
ax1.set_ylabel("Passengers waiting")
ax1.set_title("Queue length across the day  (thin lines = flight departures)")
ax1.legend(loc="upper left", frameon=False)
ax1.set_xlim(0, 28)

# --- Panel 2: active lanes across the day ---
ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
ax2.step(worst.history["time"], worst.history["lanes"], where="post", color=WORST_COLOR, lw=1.7,
          label=f"{worst.policy} (worst)")
ax2.step(best.history["time"], best.history["lanes"], where="post", color=BEST_COLOR, lw=1.7,
          label=f"{best.policy} (best)")
ax2.set_ylabel("Active lanes")
ax2.set_xlabel("Time of day (hours)")
ax2.set_title("Staffing schedule across the day")
ax2.legend(loc="upper left", frameon=False)
ax2.set_ylim(0, MAX_LANES + 1)

# --- Panel 3: cost vs. lateness trade-off, all policies ---
ax3 = fig.add_subplot(gs[2, 0])
for r in results:
    if r is best or r is worst:
        continue
    ax3.scatter(r.lane_hours, r.late_rate * 100, color=OTHER_COLOR, s=26, zorder=2)
ax3.plot([r.lane_hours for r in frontier], [r.late_rate * 100 for r in frontier],
          color="#555555", lw=1.0, ls="--", zorder=1, label="Pareto frontier")
ax3.scatter([worst.lane_hours], [worst.late_rate * 100], color=WORST_COLOR, s=150,
            edgecolor="white", linewidth=1.2, zorder=3, label=f"{worst.policy} (worst)")
ax3.scatter([best.lane_hours], [best.late_rate * 100], color=BEST_COLOR, s=150,
            edgecolor="white", linewidth=1.2, zorder=3, label=f"{best.policy} (best)")
ax3.set_xlabel("Lane-hours (staffing cost)")
ax3.set_ylabel("Late passengers (%)")
ax3.set_title("Cost vs. lateness — every policy")
ax3.legend(loc="upper right", frameon=False, fontsize=8.5)

# --- Panel 4: wait-time distribution, best vs worst ---
ax4 = fig.add_subplot(gs[2, 1])
worst_waits = np.clip(np.array(worst.waits), 0, 400)
best_waits = np.clip(np.array(best.waits), 0, 400)
bins = np.linspace(0, 400, 41)
ax4.hist(worst_waits, bins=bins, color=WORST_COLOR, alpha=0.55, label=f"{worst.policy} (worst)")
ax4.hist(best_waits, bins=bins, color=BEST_COLOR, alpha=0.75, label=f"{best.policy} (best)")
ax4.axvline(45, color="#222222", lw=1.2, ls=":", label="45-min deadline")
ax4.set_xlabel("Wait time (min, clipped at 400)")
ax4.set_ylabel("Passengers")
ax4.set_title("Wait-time distribution")
ax4.legend(loc="upper right", frameon=False, fontsize=8.5)

fig.suptitle(f"Checkpoint policy comparison — best ({best.policy}) vs. worst ({worst.policy})",
             fontsize=14, fontweight="bold", y=0.995)

fig.savefig("policy_comparison.png", dpi=160, bbox_inches="tight")
print("saved")