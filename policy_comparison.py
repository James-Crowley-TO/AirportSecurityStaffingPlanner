"""Create the main figure for the checkpoint staffing project."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from security_sim import (
    CostModel,
    FixedPolicy,
    generate_day,
    pareto_frontier,
    policy_factories,
    run_experiment,
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
BASELINE_COLOR = "#b83b3b"
OTHER_COLOR = "#a7abb0"
COST = CostModel()

# Tune on separate days, then report performance only on unseen test days.
tuned, _, summaries = run_experiment()
summary_by_name = {summary.policy: summary for summary in summaries}
best_summary = summary_by_name[str(tuned)]

fixed_summaries = [s for s in summaries if s.policy.startswith("Fixed")]
best_fixed_summary = min(fixed_summaries, key=lambda s: s.mean_score)
best_fixed_lanes = int(best_fixed_summary.policy.split()[1])
frontier = pareto_frontier(summaries)

# Use one held-out day only for the time-series illustration.
scenario = generate_day(107)
best_day = simulate(scenario, type(tuned)(tuned.target_minutes, tuned.safety_factor), service_seed=100_107)
fixed_day = simulate(scenario, FixedPolicy(best_fixed_lanes), service_seed=100_107)

print("Selected:", best_summary.policy)
print("Baseline:", best_fixed_summary.policy)

fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.25], hspace=0.5, wspace=0.28)

# --- Panel 1: queue length on a held-out day ---
ax1 = fig.add_subplot(gs[0, :])
for flight in scenario.flights:
    ax1.axvline(flight.departure, color="#eeeeee", lw=0.45, zorder=0)
ax1.plot(
    fixed_day.history["time"], fixed_day.history["queue"],
    color=BASELINE_COLOR, lw=1.7, label=f"{best_fixed_summary.policy} baseline"
)
ax1.plot(
    best_day.history["time"], best_day.history["queue"],
    color=BEST_COLOR, lw=1.8, label=f"{best_summary.policy} selected policy"
)
ax1.set_ylabel("Passengers waiting")
ax1.set_title("Held-out day: queue length (thin lines = flight departures)")
ax1.legend(loc="upper left", frameon=False)
ax1.set_xlim(-4, 28)

# --- Panel 2: staffing decisions on the same held-out day ---
ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
ax2.step(
    fixed_day.history["time"], fixed_day.history["lanes"], where="post",
    color=BASELINE_COLOR, lw=1.7, label=best_fixed_summary.policy
)
ax2.step(
    best_day.history["time"], best_day.history["lanes"], where="post",
    color=BEST_COLOR, lw=1.8, label=best_summary.policy
)
ax2.set_ylabel("Open lanes")
ax2.set_xlabel("Time relative to operating day (hours)")
ax2.set_title("Held-out day: staffing schedule")
ax2.legend(loc="upper left", frameon=False)

# --- Panel 3: Monte Carlo cost-vs-service trade-off on unseen days ---
ax3 = fig.add_subplot(gs[2, 0])
for summary in summaries:
    color = OTHER_COLOR
    size = 35
    zorder = 2
    if summary.policy == best_summary.policy:
        color, size, zorder = BEST_COLOR, 150, 4
    elif summary.policy == best_fixed_summary.policy:
        color, size, zorder = BASELINE_COLOR, 130, 3
    ax3.scatter(
        summary.mean_lane_hours, summary.mean_late_rate * 100,
        color=color, s=size, edgecolor="white" if size > 100 else None,
        linewidth=1.1 if size > 100 else 0, zorder=zorder
    )

ax3.plot(
    [s.mean_lane_hours for s in frontier],
    [s.mean_late_rate * 100 for s in frontier],
    color="#555555", lw=1.0, ls="--", label="Pareto frontier"
)
ax3.scatter([], [], color=BEST_COLOR, s=100, label="Selected policy")
ax3.scatter([], [], color=BASELINE_COLOR, s=100, label="Best fixed baseline")
ax3.set_xlabel("Mean lane-hours per day")
ax3.set_ylabel("Mean late passengers (%)")
ax3.set_title("Unseen days: staffing cost vs. service failure")
ax3.legend(loc="upper right", frameon=False, fontsize=8.5)

# --- Panel 4: out-of-sample objective distribution ---
ax4 = fig.add_subplot(gs[2, 1])
bins = np.histogram_bin_edges(
    np.array(best_summary.daily_scores + best_fixed_summary.daily_scores), bins=16
)
ax4.hist(
    best_fixed_summary.daily_scores, bins=bins, alpha=0.55,
    color=BASELINE_COLOR, label=best_fixed_summary.policy
)
ax4.hist(
    best_summary.daily_scores, bins=bins, alpha=0.72,
    color=BEST_COLOR, label=best_summary.policy
)
ax4.axvline(np.mean(best_fixed_summary.daily_scores), color=BASELINE_COLOR, lw=1.4, ls=":")
ax4.axvline(np.mean(best_summary.daily_scores), color=BEST_COLOR, lw=1.4, ls=":")
ax4.set_xlabel("Daily objective (relative cost units)")
ax4.set_ylabel("Held-out simulation days")
ax4.set_title("Out-of-sample objective distribution")
ax4.legend(loc="upper right", frameon=False, fontsize=8.5)

improvement = 100 * (best_fixed_summary.mean_score - best_summary.mean_score) / best_fixed_summary.mean_score
fig.suptitle(
    f"Airport checkpoint staffing optimizer — {improvement:.1f}% lower mean objective than best fixed staffing",
    fontsize=14, fontweight="bold", y=0.995
)
fig.text(
    0.5, 0.008,
    "Policy tuned on seeds 0–11; all reported aggregate metrics use unseen seeds 100–119. "
    "Objective = lane-hours + 25 × late passengers.",
    ha="center", fontsize=9, color="#555555"
)

fig.savefig("policy_comparison.png", dpi=170, bbox_inches="tight")
print("saved policy_comparison.png")
