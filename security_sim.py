"""Airport security checkpoint staffing under stochastic demand.

The project asks a deliberately narrow operations-research question:
    How many screening lanes should be open through the day when flight demand
    is known in advance but passenger arrivals and screening completions are not?

Policies observe only information that would plausibly be available in real time:
the current queue, current staffing, and a demand forecast derived from the flight
schedule.  They never see realized future passenger arrivals.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from statistics import mean, quantiles

import numpy as np

HOURS = 24
STEP_MINUTES = 5
MIN_LANES = 2
MAX_LANES = 40
SERVICE_RATE_PER_LANE = 2.2  # passengers / lane / minute
DEADLINE_BUFFER_MINUTES = 45
FORECAST_HORIZON_MINUTES = 30

# Passenger show-up model: hours before departure.
ARRIVAL_MEAN_HOURS = 2.0
ARRIVAL_SD_HOURS = 0.55
ARRIVAL_MIN_HOURS = 0.15
ARRIVAL_MAX_HOURS = 4.0

# These are relative cost units, not claimed dollar values.  Keeping them explicit
# makes the trade-off auditable and easy to sensitivity-test.
LANE_HOUR_COST = 1.0
LATE_PASSENGER_COST = 25.0


@dataclass(frozen=True)
class Flight:
    departure: float
    passengers: int


@dataclass(frozen=True)
class Passenger:
    arrival: float
    departure: float


@dataclass(frozen=True)
class Scenario:
    flights: list[Flight]
    passengers: list[Passenger]


@dataclass(frozen=True)
class CheckpointState:
    time: float
    queue_length: int
    active_lanes: int
    expected_arrivals: float
    upcoming_passengers: int


@dataclass(frozen=True)
class CostModel:
    lane_hour: float = LANE_HOUR_COST
    late_passenger: float = LATE_PASSENGER_COST

    def score(self, result: SimulationResult) -> float:
        return self.lane_hour * result.lane_hours + self.late_passenger * result.late_passengers


class Policy(ABC):
    @abstractmethod
    def decide(self, state: CheckpointState) -> int:
        """Return desired number of open screening lanes."""


class FixedPolicy(Policy):
    """Baseline: keep the same number of lanes open all day."""

    def __init__(self, lanes: int):
        self.lanes = lanes

    def decide(self, state: CheckpointState) -> int:
        return self.lanes

    def __str__(self) -> str:
        return f"Fixed {self.lanes}"


class QueueThresholdPolicy(Policy):
    """Reactive baseline that uses only the observed queue."""

    def __init__(self, low: int = 20, high: int = 60, medium_lanes: int = 6):
        self.low = low
        self.high = high
        self.medium_lanes = medium_lanes

    def decide(self, state: CheckpointState) -> int:
        if state.queue_length >= self.high:
            return MAX_LANES
        if state.queue_length >= self.low:
            return self.medium_lanes
        return MIN_LANES

    def __str__(self) -> str:
        return "Queue threshold"


class FlightBankPolicy(Policy):
    """Schedule-driven baseline using only near-term flight volume."""

    def __init__(self, medium_threshold: int = 250, high_threshold: int = 600):
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold

    def decide(self, state: CheckpointState) -> int:
        if state.upcoming_passengers >= self.high_threshold:
            return 12
        if state.upcoming_passengers >= self.medium_threshold:
            return 6
        return MIN_LANES

    def __str__(self) -> str:
        return "Flight-bank forecast"


class TargetClearancePolicy(Policy):
    """Main policy family.

    Convert observed queue + forecast arrivals into required throughput.  The two
    parameters are intentionally interpretable and small enough to tune by grid
    search without turning the project into an ML exercise.
    """

    def __init__(self, target_minutes: int = 30, safety_factor: float = 1.15):
        self.target_minutes = target_minutes
        self.safety_factor = safety_factor

    def decide(self, state: CheckpointState) -> int:
        forecast_for_target = state.expected_arrivals * (self.target_minutes / FORECAST_HORIZON_MINUTES)
        demand = state.queue_length + self.safety_factor * forecast_for_target
        capacity_per_lane = SERVICE_RATE_PER_LANE * self.target_minutes
        return math.ceil(demand / capacity_per_lane) if demand > 0 else MIN_LANES

    def __str__(self) -> str:
        return f"Target-clear {self.target_minutes}m x{self.safety_factor:.2f}"


@dataclass
class SimulationResult:
    policy: str
    late_passengers: int
    total_passengers: int
    lane_hours: float
    average_wait: float
    p95_wait: float
    maximum_queue: int
    history: dict[str, list] = field(default_factory=dict)
    waits: list[float] = field(default_factory=list)

    @property
    def late_rate(self) -> float:
        return self.late_passengers / self.total_passengers if self.total_passengers else 0.0

    def score(self, cost_model: CostModel = CostModel()) -> float:
        return cost_model.score(self)


@dataclass(frozen=True)
class PolicySummary:
    policy: str
    mean_score: float
    mean_late_rate: float
    mean_lane_hours: float
    mean_p95_wait: float
    mean_max_queue: float
    daily_scores: tuple[float, ...]


PolicyFactory = Callable[[], Policy]


def fixed_lane_options() -> tuple[int, ...]:
    """Compact fixed-staffing sweep used as a simple benchmark frontier."""
    return (2, 4, 6, 8, 10, 12, 16, 20, 30, 40)


FIXED_LANE_OPTIONS = fixed_lane_options()


@lru_cache(maxsize=None)
def generate_day(seed: int = 7) -> Scenario:
    """Generate one reproducible day of flights and passenger show-up times.

    Flight departures follow a piecewise Poisson process with morning/evening banks.
    Conditional on a flight, every passenger independently chooses a show-up time
    from the same truncated normal distribution.  There is deliberately no passenger
    heterogeneity in screening difficulty.
    """
    rng = random.Random(seed)
    flights: list[Flight] = []

    for hour in range(HOURS):
        flight_rate = 1.0 if hour < 6 or hour >= 22 else 3.0
        if 7 <= hour < 10 or 16 <= hour < 20:
            flight_rate = 7.0

        for _ in range(_poisson(rng, flight_rate)):
            departure = hour + rng.random()
            passengers = rng.randint(40, 400)
            flights.append(Flight(departure, passengers))

    flights.sort(key=lambda f: f.departure)

    passengers: list[Passenger] = []
    for flight in flights:
        for _ in range(flight.passengers):
            hours_early = _truncated_normal(
                rng,
                ARRIVAL_MEAN_HOURS,
                ARRIVAL_SD_HOURS,
                ARRIVAL_MIN_HOURS,
                ARRIVAL_MAX_HOURS,
            )
            passengers.append(Passenger(flight.departure - hours_early, flight.departure))

    passengers.sort(key=lambda p: p.arrival)
    return Scenario(flights, passengers)


def expected_arrivals_from_schedule(
    flights: Iterable[Flight],
    time: float,
    horizon_minutes: int = FORECAST_HORIZON_MINUTES,
) -> float:
    """Expected passenger arrivals in (time, time + horizon] from the flight schedule.

    This is a forecast, not an oracle: it integrates the assumed passenger show-up
    distribution and does not inspect the realized passenger list.
    """
    horizon = horizon_minutes / 60.0
    expected = 0.0

    for flight in flights:
        # arrival = departure - hours_early
        lower_early = flight.departure - (time + horizon)
        upper_early = flight.departure - time
        probability = _truncated_normal_interval_probability(lower_early, upper_early)
        expected += flight.passengers * probability

    return expected


def simulate(
    scenario: Scenario,
    policy: Policy,
    service_seed: int = 10_000,
) -> SimulationResult:
    """Simulate one day under a staffing policy.

    Screening completions are stochastic and homogeneous.  At every step, each
    potential lane receives an independent Poisson number of service completions;
    a policy opening N lanes uses the first N draws.  Reusing service_seed across
    policies therefore provides common random numbers and reduces comparison noise.
    """
    arrivals = scenario.passengers
    waiting: list[Passenger] = []
    waits: list[float] = []
    late_passengers = 0
    lane_hours = 0.0
    maximum_queue = 0
    history = {
        "time": [],
        "queue": [],
        "lanes": [],
        "forecast_arrivals": [],
        "flight_passengers_2h": [],
    }

    service_rng = np.random.default_rng(service_seed)
    next_arrival = 0
    time = min(0.0, arrivals[0].arrival if arrivals else 0.0)
    end_time = HOURS + ARRIVAL_MAX_HOURS
    active_lanes = MIN_LANES
    step_hours = STEP_MINUTES / 60.0

    while time <= end_time or waiting or next_arrival < len(arrivals):
        while next_arrival < len(arrivals) and arrivals[next_arrival].arrival <= time:
            waiting.append(arrivals[next_arrival])
            next_arrival += 1

        upcoming_2h = [
            flight for flight in scenario.flights
            if time <= flight.departure <= time + 2.0
        ]
        expected = expected_arrivals_from_schedule(
            scenario.flights, time, FORECAST_HORIZON_MINUTES
        )
        upcoming_passengers = sum(flight.passengers for flight in upcoming_2h)

        state = CheckpointState(
            time=time,
            queue_length=len(waiting),
            active_lanes=active_lanes,
            expected_arrivals=expected,
            upcoming_passengers=upcoming_passengers,
        )
        desired = policy.decide(state)
        active_lanes = max(MIN_LANES, min(MAX_LANES, int(desired)))

        # Generate service shocks for all possible lanes, even if closed.  This keeps
        # the random environment aligned when comparing different policies.
        per_lane_completions = service_rng.poisson(
            SERVICE_RATE_PER_LANE * STEP_MINUTES, size=MAX_LANES
        )
        service_capacity = int(per_lane_completions[:active_lanes].sum())
        service_count = min(len(waiting), service_capacity)

        for passenger in waiting[:service_count]:
            wait_hours = max(0.0, time - passenger.arrival)
            waits.append(wait_hours * 60.0)
            deadline = passenger.departure - DEADLINE_BUFFER_MINUTES / 60.0
            # Attribute a service failure to the checkpoint only if the passenger
            # reached security before the cutoff but screening finished after it.
            if passenger.arrival <= deadline < time:
                late_passengers += 1
        del waiting[:service_count]

        lane_hours += active_lanes * step_hours
        maximum_queue = max(maximum_queue, len(waiting))
        history["time"].append(time)
        history["queue"].append(len(waiting))
        history["lanes"].append(active_lanes)
        history["forecast_arrivals"].append(expected)
        history["flight_passengers_2h"].append(upcoming_passengers)

        time += step_hours

    waits.sort()
    p95 = quantiles(waits, n=20)[18] if len(waits) >= 20 else (max(waits) if waits else 0.0)
    return SimulationResult(
        policy=str(policy),
        late_passengers=late_passengers,
        total_passengers=len(arrivals),
        lane_hours=lane_hours,
        average_wait=mean(waits) if waits else 0.0,
        p95_wait=p95,
        maximum_queue=maximum_queue,
        history=history,
        waits=waits,
    )


def evaluate_policy(
    factory: PolicyFactory,
    seeds: Iterable[int],
    cost_model: CostModel = CostModel(),
) -> PolicySummary:
    """Monte Carlo evaluation over independent demand days."""
    results = [
        simulate(generate_day(seed), factory(), service_seed=100_000 + seed)
        for seed in seeds
    ]
    scores = tuple(result.score(cost_model) for result in results)
    return PolicySummary(
        policy=results[0].policy,
        mean_score=mean(scores),
        mean_late_rate=mean(result.late_rate for result in results),
        mean_lane_hours=mean(result.lane_hours for result in results),
        mean_p95_wait=mean(result.p95_wait for result in results),
        mean_max_queue=mean(result.maximum_queue for result in results),
        daily_scores=scores,
    )


def tune_target_clearance(
    training_seeds: Iterable[int],
    cost_model: CostModel = CostModel(),
) -> tuple[TargetClearancePolicy, list[PolicySummary]]:
    """Tune a small, interpretable policy grid on training simulation days."""
    candidates: list[PolicySummary] = []
    parameter_grid = [
        (target_minutes, safety_factor)
        for target_minutes in (15, 20, 30, 40)
        for safety_factor in (1.00, 1.15, 1.30)
    ]

    for target_minutes, safety_factor in parameter_grid:
        candidates.append(
            evaluate_policy(
                lambda tm=target_minutes, sf=safety_factor: TargetClearancePolicy(tm, sf),
                training_seeds,
                cost_model,
            )
        )

    best = min(candidates, key=lambda summary: summary.mean_score)
    # Parse from the winning summary only to avoid adding hidden state to PolicySummary.
    parts = best.policy.split()
    target_minutes = int(parts[1].removesuffix("m"))
    safety_factor = float(parts[2].removeprefix("x"))
    return TargetClearancePolicy(target_minutes, safety_factor), candidates


def pareto_frontier(summaries: Iterable[PolicySummary]) -> list[PolicySummary]:
    """Non-dominated policies in mean lane-hours vs. mean late-rate space."""
    ordered = sorted(summaries, key=lambda s: (s.mean_lane_hours, s.mean_late_rate))
    frontier: list[PolicySummary] = []
    best_late = math.inf
    for summary in ordered:
        if summary.mean_late_rate < best_late:
            frontier.append(summary)
            best_late = summary.mean_late_rate
    return frontier


def policy_factories(tuned: TargetClearancePolicy | None = None) -> list[PolicyFactory]:
    factories: list[PolicyFactory] = [
        lambda lanes=lanes: FixedPolicy(lanes) for lanes in FIXED_LANE_OPTIONS
    ]
    factories += [
        QueueThresholdPolicy,
        FlightBankPolicy,
    ]
    if tuned is not None:
        factories.append(
            lambda: TargetClearancePolicy(tuned.target_minutes, tuned.safety_factor)
        )
    return factories


def run_experiment(
    training_seeds: range = range(0, 12),
    test_seeds: range = range(100, 120),
    cost_model: CostModel = CostModel(),
):
    """Tune on one set of simulated days, then compare policies on unseen days."""
    tuned, tuning_results = tune_target_clearance(training_seeds, cost_model)
    summaries = [
        evaluate_policy(factory, test_seeds, cost_model)
        for factory in policy_factories(tuned)
    ]
    summaries.sort(key=lambda s: s.mean_score)

    print(f"Tuned policy: {tuned}")
    print(
        f"Cost model: {cost_model.lane_hour:g} per lane-hour + "
        f"{cost_model.late_passenger:g} per late passenger\n"
    )
    for summary in summaries:
        print(
            f"{summary.policy:24} "
            f"score={summary.mean_score:8.1f}  "
            f"late={summary.mean_late_rate:7.3%}  "
            f"lane-hours={summary.mean_lane_hours:7.1f}  "
            f"p95 wait={summary.mean_p95_wait:6.1f}m"
        )

    return tuned, tuning_results, summaries


def _poisson(rng: random.Random, mean_value: float) -> int:
    """Knuth Poisson sampler; sufficient for the deliberately small step means here."""
    threshold = math.exp(-mean_value)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def _truncated_normal(
    rng: random.Random,
    mu: float,
    sigma: float,
    low: float,
    high: float,
) -> float:
    while True:
        value = rng.gauss(mu, sigma)
        if low <= value <= high:
            return value


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def _truncated_normal_interval_probability(low: float, high: float) -> float:
    low = max(low, ARRIVAL_MIN_HOURS)
    high = min(high, ARRIVAL_MAX_HOURS)
    if high <= low:
        return 0.0

    denominator = _normal_cdf(ARRIVAL_MAX_HOURS, ARRIVAL_MEAN_HOURS, ARRIVAL_SD_HOURS) - _normal_cdf(
        ARRIVAL_MIN_HOURS, ARRIVAL_MEAN_HOURS, ARRIVAL_SD_HOURS
    )
    numerator = _normal_cdf(high, ARRIVAL_MEAN_HOURS, ARRIVAL_SD_HOURS) - _normal_cdf(
        low, ARRIVAL_MEAN_HOURS, ARRIVAL_SD_HOURS
    )
    return numerator / denominator


if __name__ == "__main__":
    run_experiment()
