"""Airport security checkpoint capacity planning under stochastic demand."""

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from statistics import mean, quantiles

HOURS = 24
STEP_MINUTES = 5
MIN_LANES = 2
MAX_LANES = 40
SERVICE_RATE_PER_LANE = 2.2
MINUTES_BEFORE_DEPARTURE = 45


def fixed_lane_options(max_lanes: int, min_lanes: int = MIN_LANES) -> tuple[int, ...]:
    """Return a compact fixed-lane sweep whose spacing grows with system size."""
    if max_lanes <= 12:
        step = 2
    elif max_lanes <= 24:
        step = 3
    elif max_lanes <= 32:
        step = 4
    else:
        step = 5

    options = list(range(min_lanes, max_lanes + 1, step))
    if max_lanes not in options:
        options.append(max_lanes)
    return tuple(options)


FIXED_LANE_OPTIONS = fixed_lane_options(MAX_LANES)


@dataclass(frozen=True)
class Flight:
    departure: float
    capacity: int


@dataclass(frozen=True)
class Passenger:
    arrival: float
    departure: float


@dataclass
class Scenario:
    flights: list[Flight]
    passengers: list[Passenger]


@dataclass
class CheckpointState:
    time: float
    queue_length: int
    active_lanes: int
    upcoming_flights: list[Flight]
    passengers_waiting: int
    expected_arrivals: int = 0


class Policy(ABC):
    @abstractmethod
    def decide(self, state: CheckpointState) -> int:
        """Return desired number of open lanes."""


class FixedPolicy(Policy):
    def __init__(self, lanes: int):
        self.lanes = lanes

    def decide(self, state: CheckpointState) -> int:
        return self.lanes

    def __str__(self) -> str:
        return f"Fixed {self.lanes} lanes"


class QueueThresholdPolicy(Policy):
    def decide(self, state: CheckpointState) -> int:
        if state.queue_length > 20:
            return MAX_LANES
        if state.queue_length > 10:
            return 4
        return MIN_LANES

    def __str__(self) -> str:
        return "Queue threshold"


class LookaheadPolicy(Policy):
    def __init__(self, lookahead_hours: float = 2.0):
        self.lookahead_hours = lookahead_hours

    def decide(self, state: CheckpointState) -> int:
        demand = state.expected_arrivals + state.queue_length
        minutes_until_peak = min(
            (flight.departure - state.time) * 60 for flight in state.upcoming_flights
        ) if state.upcoming_flights else math.inf
        if demand > 45 or minutes_until_peak < 60 and demand > 20:
            return MAX_LANES
        if demand > 20 or minutes_until_peak < 90:
            return 4
        return MIN_LANES

    def __str__(self) -> str:
        return f"Lookahead {self.lookahead_hours:g}h"


class HysteresisPolicy(Policy):
    """Ramps lanes up quickly under load, but only backs off once demand has been
    comfortably below a separate, lower threshold -- avoiding the rapid open/close
    thrashing that a single-threshold policy exhibits near the boundary."""

    def __init__(self, up_threshold: int = 15, down_threshold: int = 5, ramp_up: int = 2):
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.ramp_up = ramp_up
        self.current = MIN_LANES

    def decide(self, state: CheckpointState) -> int:
        demand = state.queue_length + state.expected_arrivals
        if demand > self.up_threshold:
            self.current = min(MAX_LANES, self.current + self.ramp_up)
        elif demand < self.down_threshold:
            self.current = max(MIN_LANES, self.current - 1)
        return self.current

    def __str__(self) -> str:
        return "Hysteresis"


class TargetClearancePolicy(Policy):
    """Directly converts forecasted demand into required throughput: sizes lanes so
    that the current queue plus near-term expected arrivals would clear within a
    target window, rather than relying on hand-tuned step thresholds."""

    def __init__(self, target_minutes: float = 30.0):
        self.target_minutes = target_minutes

    def decide(self, state: CheckpointState) -> int:
        demand = state.queue_length + state.expected_arrivals
        if demand == 0:
            return MIN_LANES
        required = demand / (SERVICE_RATE_PER_LANE * self.target_minutes)
        return math.ceil(required)

    def __str__(self) -> str:
        return f"Target-clear {self.target_minutes:g}min"


class FlightBankPolicy(Policy):
    """Uses flight capacity as a demand forecast before passengers reach the queue."""

    def __init__(self, forecast_fraction: float = 0.45):
        self.forecast_fraction = forecast_fraction

    def decide(self, state: CheckpointState) -> int:
        flight_capacity = sum(flight.capacity for flight in state.upcoming_flights)
        forecast = state.queue_length + state.expected_arrivals
        forecast += self.forecast_fraction * flight_capacity
        if forecast > 180:
            return MAX_LANES
        if forecast > 80:
            return 6
        if forecast > 35:
            return 4
        return MIN_LANES

    def __str__(self) -> str:
        return "Flight-bank forecast"


class DeadlineAwarePolicy(Policy):
    """Adds capacity when a departure bank is close enough to create late passengers."""

    def decide(self, state: CheckpointState) -> int:
        urgent_capacity = sum(
            flight.capacity for flight in state.upcoming_flights
            if flight.departure - state.time <= 1.0
        )
        demand = state.queue_length + state.expected_arrivals
        if urgent_capacity > 250 or (urgent_capacity > 100 and demand > 25):
            return MAX_LANES
        if urgent_capacity > 100 or demand > 45:
            return 6
        if urgent_capacity > 0 or demand > 20:
            return 4
        return MIN_LANES

    def __str__(self) -> str:
        return "Deadline-aware"


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
        return self.late_passengers / self.total_passengers

    @property
    def objective(self) -> float:
        return self.late_passengers + self.lane_hours


def generate_day(seed: int = 7) -> Scenario:
    """Generate flight banks and passenger arrivals using one reproducible seed."""
    rng = random.Random(seed)
    flights: list[Flight] = []
    for hour in range(HOURS):
        rate = 1.0 if hour < 6 or hour >= 22 else 3.0
        if 7 <= hour < 10 or 16 <= hour < 20:
            rate = 7.0
        for _ in range(_poisson(rng, rate)):
            departure = hour + rng.random()
            flights.append(Flight(departure, rng.randint(40, 400)))
    flights.sort(key=lambda flight: flight.departure)

    passengers = []
    for flight in flights:
        for _ in range(flight.capacity):
            hours_early = min(4.0, max(0.15, rng.gauss(2.0, 0.55)))
            passengers.append(Passenger(flight.departure - hours_early, flight.departure))
    passengers.sort(key=lambda passenger: passenger.arrival)
    return Scenario(flights, passengers)


def _poisson(rng: random.Random, mean_value: float) -> int:
    threshold = math.exp(-mean_value)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def simulate(scenario: Scenario, policy: Policy) -> SimulationResult:
    arrivals = sorted(scenario.passengers, key=lambda passenger: passenger.arrival)
    waiting: list[Passenger] = []
    waits: list[float] = []
    late_passengers = 0
    lane_hours = 0.0
    maximum_queue = 0
    history = {"time": [], "queue": [], "lanes": [], "departures": [], "arrivals": []}
    next_arrival = 0
    time = 0.0
    end_time = HOURS + 4.0

    while time <= end_time or waiting:
        while next_arrival < len(arrivals) and arrivals[next_arrival].arrival <= time:
            waiting.append(arrivals[next_arrival])
            next_arrival += 1
        upcoming = [flight for flight in scenario.flights
                    if time <= flight.departure <= time + 2.0]
        expected = sum(1 for passenger in arrivals[next_arrival:]
                       if time < passenger.arrival <= time + 1.0)
        state = CheckpointState(time, len(waiting), MIN_LANES, upcoming, len(waiting), expected)
        lanes = max(MIN_LANES, min(MAX_LANES, policy.decide(state)))
        service_count = min(len(waiting), int(lanes * SERVICE_RATE_PER_LANE * STEP_MINUTES))
        for passenger in waiting[:service_count]:
            wait = max(0.0, time - passenger.arrival)
            waits.append(wait * 60)
            if time > passenger.departure - MINUTES_BEFORE_DEPARTURE / 60:
                late_passengers += 1
        del waiting[:service_count]

        lane_hours += lanes * STEP_MINUTES / 60
        maximum_queue = max(maximum_queue, len(waiting))
        history["time"].append(time)
        history["queue"].append(len(waiting))
        history["lanes"].append(lanes)
        history["departures"].append(sum(1 for flight in scenario.flights
                                          if time <= flight.departure < time + STEP_MINUTES / 60))
        history["arrivals"].append(expected)
        time += STEP_MINUTES / 60

    waits.sort()
    p95 = quantiles(waits, n=20)[18] if len(waits) >= 20 else (max(waits) if waits else 0.0)
    return SimulationResult(str(policy), late_passengers, len(arrivals), lane_hours,
                            mean(waits) if waits else 0.0, p95, maximum_queue, history, waits)


def pareto_frontier(results: list[SimulationResult]) -> list[SimulationResult]:
    frontier = []
    for result in sorted(results, key=lambda item: item.lane_hours):
        if not frontier or result.late_passengers < frontier[-1].late_passengers:
            frontier.append(result)
    return frontier


def weighted_best(results: list[SimulationResult], lambda_late: float,
                  mu_lane_hours: float) -> SimulationResult:
    return min(results, key=lambda result: (
        lambda_late * result.late_passengers + mu_lane_hours * result.lane_hours
    ))


def run_experiment(seed: int = 7, show_plot: bool = True):
    scenario = generate_day(seed)
    policies = [FixedPolicy(lanes) for lanes in FIXED_LANE_OPTIONS]
    policies += [QueueThresholdPolicy(), LookaheadPolicy(),
                 HysteresisPolicy(), TargetClearancePolicy(),
                 FlightBankPolicy(), DeadlineAwarePolicy()]
    results = [simulate(scenario, policy) for policy in policies]
    frontier = pareto_frontier(results)

    print(f"Generated {len(scenario.flights)} flights and {len(scenario.passengers)} passengers")
    for result in results:
        print(f"{result.policy:20} late={result.late_rate:6.2%} "
              f"lane-hours={result.lane_hours:7.1f} p95-wait={result.p95_wait:5.1f} min")
    print("Pareto frontier:", ", ".join(result.policy for result in frontier))
    for ratio in (0.1, 1.0, 10.0, 100.0):
        best = weighted_best(results, lambda_late=ratio, mu_lane_hours=1.0)
        print(f"lambda/mu={ratio:5.1f}: {best.policy}")
    return results, frontier


if __name__ == "__main__":
    run_experiment(show_plot=False)