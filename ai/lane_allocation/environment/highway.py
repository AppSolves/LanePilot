import random as rd
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .vehicle import Vehicle


class HighwayEnv(gym.Env):
    """
    Multi-lane highway environment for RL-based lane allocation.
    Uses Intelligent Driver Model (IDM) for realistic car-following.
    Optimizes for traffic flow, minimal stop-and-go, and safety.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__()
        if config is None:
            config = {}
        self.num_lanes = config.get("num_lanes", 3)
        self.road_length = config.get("road_length", 1000.0)  # meters
        self.dt = config.get("dt", 0.2)
        self.max_episode_steps = config.get("max_episode_steps", 300)
        self.spawn_rate = config.get(
            "spawn_rate", 0.5
        )  # vehicles per second (tune from video)
        self.max_speed = config.get("max_speed", 33.33)  # 120 km/h in m/s ~ 33.33
        self.min_speed = config.get("min_speed", 8.33)  # 30 km/h

        # Multi-vehicle centralized control
        self.multi_vehicle_control = config.get("multi_vehicle_control", False)
        self.max_vehicles = config.get("max_vehicles", 15)
        self.initial_vehicle_count = config.get("initial_vehicle_count", 6)
        self.ego_index = None  # Only used in single-vehicle mode

        # Rendering
        self.render_mode = config.get("render_mode", None)
        self.renderer = None
        self.render_config = config.get("visualization", {})
        # Track actions: single value for single-vehicle, array for multi-vehicle
        self.last_action = (
            0
            if not self.multi_vehicle_control
            else np.zeros(self.max_vehicles, dtype=np.int32)
        )

        # Validate configuration
        if self.max_speed <= self.min_speed:
            raise ValueError(
                f"max_speed ({self.max_speed}) must be greater than min_speed ({self.min_speed})"
            )
        if self.num_lanes < 1:
            raise ValueError(f"num_lanes must be >= 1, got {self.num_lanes}")
        if self.spawn_rate < 0:
            raise ValueError(f"spawn_rate must be >= 0, got {self.spawn_rate}")

        # Action space:
        # - Single-vehicle mode: Discrete(3) - control one ego vehicle
        # - Multi-vehicle mode: MultiDiscrete([3]*max_vehicles) - control all vehicles
        if self.multi_vehicle_control:
            self.action_space = spaces.MultiDiscrete([3] * self.max_vehicles)
        else:
            self.action_space = spaces.Discrete(3)

        # Observation space:
        # - Single-vehicle mode: local view (15 features)
        # - Multi-vehicle mode: global state (max_vehicles * 5 features + fixed global features)
        if self.multi_vehicle_control:
            # Per-vehicle: [lane, position, speed, acceleration, target_speed] = 5 features
            # Global features (fixed size for any num_lanes):
            #   - avg_speed_all (1)
            #   - speed_std_all (1)
            #   - num_vehicles_norm (1)
            #   - num_lanes_norm (1) - NEW: tells network how many lanes are active
            #   - lane_balance_score (1) - variance in lane distribution
            #   - collision_rate (1)
            # Total global: 6 features (fixed)
            obs_len = self.max_vehicles * 5 + 6
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32
            )
        else:
            # Single-vehicle: [own_lane, own_speed, own_acc, gap_front, rel_speed_front,
            #                  gap_left_front, gap_left_back, gap_right_front, gap_right_back,
            #                  lane_densities (3), avg_speed_per_lane (3)]
            obs_len = 15
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32
            )

        # internal state
        self.vehicles: list[Vehicle] = []
        self.step_count = 0
        self.total_reward = 0.0

        # metrics for evaluation
        self.speed_history = []
        self.lane_change_count = 0
        self.lane_changes_this_step = (
            0  # Track lane changes in current step for immediate penalty
        )
        self.hard_braking_count = 0
        self.collision_count = 0

        self.seed()

    def seed(self, seed: int | None = None):
        rd.seed(seed)
        np.random.seed(seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        self.vehicles = []
        self.step_count = 0
        self.total_reward = 0.0
        self.speed_history = []
        self.lane_change_count = 0
        self.lane_changes_this_step = 0
        self.hard_braking_count = 0
        self.collision_count = 0

        if self.multi_vehicle_control:
            # Multi-vehicle mode: spawn initial_vehicle_count vehicles, ALL controlled by RL
            for i in range(self.initial_vehicle_count):
                lane = rd.randrange(self.num_lanes)
                pos = rd.uniform(0, self.road_length * 0.6)
                # Varied speed distribution to create different vehicle classes
                speed = rd.uniform(self.min_speed, self.max_speed)
                # Desired speed determines if vehicle should be in fast/slow lane
                desired_speed = rd.uniform(self.min_speed, self.max_speed)
                v = Vehicle(
                    lane=lane,
                    position=pos,
                    speed=speed,
                    max_speed=self.max_speed,
                    desired_speed=desired_speed,
                    is_controlled=True,  # ALL vehicles are RL-controlled
                )
                self.vehicles.append(v)
            # Sort by position to avoid immediate collisions
            self.vehicles.sort(key=lambda x: x.position)
        else:
            # Single-vehicle mode: spawn background traffic + one ego vehicle
            initial_count = int(self.spawn_rate * 20)
            for i in range(initial_count):
                lane = rd.randrange(self.num_lanes)
                pos = rd.uniform(0, self.road_length * 0.6)
                # Realistic speed distribution: mostly around speed limit
                speed = np.clip(
                    np.random.normal(self.max_speed * 0.85, 5.0),
                    self.min_speed,
                    self.max_speed,
                )
                v = Vehicle(
                    lane=lane,
                    position=pos,
                    speed=speed,
                    max_speed=self.max_speed,
                    desired_speed=speed,
                    is_controlled=False,
                )
                self.vehicles.append(v)

            # Sort by position to avoid immediate collisions
            self.vehicles.sort(key=lambda x: x.position)

            # Create ego vehicle (RL-controlled)
            ego_speed = rd.uniform(self.min_speed, self.max_speed)
            ego = Vehicle(
                lane=1,
                position=10.0,
                speed=ego_speed,
                max_speed=self.max_speed,
                desired_speed=self.max_speed * 0.9,  # wants to go fast
                is_controlled=True,
            )
            self.vehicles.insert(0, ego)
            self.ego_index = 0

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        self.lane_changes_this_step = 0  # Reset counter for this step

        if self.multi_vehicle_control:
            # Multi-vehicle mode: apply actions to ALL vehicles
            # Capture state BEFORE any modifications
            num_vehicles_before = len(self.vehicles)
            old_is_changing = [v.is_changing_lane for v in self.vehicles]

            # Apply actions to existing vehicles (action is array of size max_vehicles)
            actions_executed = []
            for i in range(min(num_vehicles_before, self.max_vehicles)):
                if i < len(action):
                    executed = self._apply_action(self.vehicles[i], action[i])
                    actions_executed.append(action[i] if executed else 0)
                    # Track lane changes when they START (not when they complete)
                    # Safe access since i < num_vehicles_before
                    if (
                        i < len(old_is_changing)
                        and not old_is_changing[i]
                        and self.vehicles[i].is_changing_lane
                    ):
                        self.lane_change_count += 1
                        self.lane_changes_this_step += 1

            # Store actions for visualization
            self.last_action = np.array(
                actions_executed + [0] * (self.max_vehicles - len(actions_executed)),
                dtype=np.int32,
            )

            # Update all vehicles using IDM for longitudinal control
            self._update_vehicles_idm()

            # Update positions
            self._update_positions()

            # In multi-vehicle mode, spawn new vehicles up to max_vehicles
            if (
                len(self.vehicles) < self.max_vehicles
                and rd.random() < self.spawn_rate * self.dt
            ):
                self._spawn_vehicle_controlled()

        else:
            # Single-vehicle mode: apply action only to ego vehicle
            if self.ego_index is None:
                raise RuntimeError("Environment must be reset before stepping.")

            ego = self.vehicles[self.ego_index]
            old_lane = ego.lane

            # Apply ego action (lane change only) and track if it was actually executed
            action_executed = self._apply_action(ego, action)

            # Store action for visualization (0 if not executed)
            self.last_action = action if action_executed else 0

            if ego.lane != old_lane:
                self.lane_change_count += 1

            # Update all vehicles using IDM for longitudinal control
            self._update_vehicles_idm()

            # Update positions based on velocities
            self._update_positions()

            # Spawn new vehicles probabilistically
            if rd.random() < self.spawn_rate * self.dt:
                self._spawn_vehicle()

        # Track metrics
        avg_speed = sum(v.speed for v in self.vehicles) / max(1, len(self.vehicles))
        self.speed_history.append(avg_speed)

        obs = self._get_obs()
        reward = self._compute_reward()
        self.total_reward += reward

        done = self.step_count >= self.max_episode_steps
        truncated = False

        info = {
            "avg_speed": avg_speed,
            "num_vehicles": len(self.vehicles),
            "lane_changes": self.lane_change_count,
            "hard_braking": self.hard_braking_count,
            "collisions": self.collision_count,
            "reward": reward,
            "step": self.step_count,
            "max_steps": self.max_episode_steps,
            "num_lanes": self.num_lanes,  # Debug: verify num_lanes context
        }

        return obs, reward, done, truncated, info

    def _apply_action(self, ego, action):
        """Apply lane change action. Actions: 0=keep, 1=left, 2=right

        Returns:
            bool: True if action was executed, False if blocked
        """
        # STRICT cooldown enforcement - must wait full duration even after transition completes
        # This prevents oscillation by ensuring vehicles wait before making another lane change
        if ego.lane_change_cooldown > 0:
            return False

        # Also block if currently transitioning
        if ego.is_changing_lane:
            return False

        if action == 1 and ego.lane > 0:
            # Check safety for left lane change
            if self._is_lane_change_safe(ego, ego.lane - 1):
                # Start smooth lane change transition
                ego.source_lane = ego.lane
                ego.target_lane = ego.lane - 1
                ego.is_changing_lane = True
                ego.lane_transition_progress = 0.0
                # Set cooldown to cover transition (1.5s) + mandatory wait (5s) = 6.5s total
                ego.lane_change_cooldown = (
                    9  # Dramatically increased to prevent oscillation
                )
                return True
        elif action == 2 and ego.lane < self.num_lanes - 1:
            # Check safety for right lane change
            if self._is_lane_change_safe(ego, ego.lane + 1):
                # Start smooth lane change transition
                ego.source_lane = ego.lane
                ego.target_lane = ego.lane + 1
                ego.is_changing_lane = True
                ego.lane_transition_progress = 0.0
                # Set cooldown to cover transition (1.5s) + mandatory wait (5s) = 6.5s total
                ego.lane_change_cooldown = (
                    9  # Dramatically increased to prevent oscillation
                )
                return True
        # action 0: keep lane - do nothing
        return action == 0  # Return True for keep lane, False for blocked lane change

    def _is_lane_change_safe(self, ego, target_lane):
        """
        Enhanced safety check for lane changes.
        Checks both front and rear gaps in target lane.
        """
        min_front_gap = 25.0  # meters (increased from 20m)
        min_rear_gap = 20.0  # meters (increased from 15m)

        for v in self.vehicles:
            if v is ego:
                continue

            # Check vehicles in target lane, transitioning to it, or transitioning from it
            is_relevant = (
                v.lane == target_lane  # Currently in target lane
                or (
                    v.is_changing_lane and v.target_lane == target_lane
                )  # Moving to target lane
                or (
                    v.is_changing_lane and v.source_lane == target_lane
                )  # Leaving target lane (still partially there)
            )

            if not is_relevant:
                continue

            distance = v.position - ego.position

            # Vehicle ahead in target lane
            if distance > 0 and distance < min_front_gap:
                return False

            # Vehicle behind in target lane
            if distance < 0 and abs(distance) < min_rear_gap:
                return False

        return True

    def _update_vehicles_idm(self):
        """
        Update vehicle accelerations using Intelligent Driver Model (IDM).
        IDM creates realistic car-following behavior and stop-and-go dynamics.
        """
        for v in self.vehicles:
            # Decrease lane change cooldown
            if v.lane_change_cooldown > 0:
                v.lane_change_cooldown = max(0.0, v.lane_change_cooldown - self.dt)

            front = self._vehicle_in_front(v)

            # IDM acceleration calculation
            if front is None or front.position <= v.position:
                # Free road - accelerate towards desired speed
                acc_free = v.max_acceleration * (1.0 - (v.speed / v.desired_speed) ** 4)
                v.acceleration = acc_free
            else:
                # Calculate desired following distance
                gap = front.position - v.position - v.length
                speed_diff = v.speed - front.speed

                # Emergency braking if too close
                if gap < v.length * 0.5:  # Less than half a car length
                    v.acceleration = (
                        -v.comfortable_deceleration * 2.0
                    )  # Double braking force
                    continue

                # IDM desired gap
                s_star = (
                    v.min_spacing
                    + v.time_headway * v.speed
                    + (v.speed * speed_diff)
                    / (2 * np.sqrt(v.max_acceleration * v.comfortable_deceleration))
                )

                # IDM acceleration (prevent division by very small gap)
                acc_free = v.max_acceleration * (1.0 - (v.speed / v.desired_speed) ** 4)
                acc_interaction = (
                    -v.max_acceleration * (s_star / max(gap, v.length * 0.5)) ** 2
                )

                v.acceleration = acc_free + acc_interaction

            # Clamp acceleration
            v.acceleration = np.clip(
                v.acceleration,
                -v.comfortable_deceleration * 2.0,  # Allow stronger emergency braking
                v.max_acceleration,
            )

            # Track hard braking events
            if v.acceleration < -v.comfortable_deceleration * 0.8:
                self.hard_braking_count += 1

    def _update_positions(self):
        """Update positions based on current speeds and accelerations."""
        for v in self.vehicles:
            # Update speed: v = v0 + a*dt
            v.speed = max(0.0, min(v.speed + v.acceleration * self.dt, v.max_speed))

            # Update position: s = s0 + v*dt + 0.5*a*dt^2
            v.position += v.speed * self.dt + 0.5 * v.acceleration * self.dt**2

            # Update lane transition progress for smooth animation
            if v.is_changing_lane:
                v.lane_transition_progress += self.dt / v.lane_transition_duration
                if v.lane_transition_progress >= 1.0:
                    # Complete the lane change
                    v.lane = v.target_lane
                    v.is_changing_lane = False
                    v.lane_transition_progress = 0.0

        # Enforce minimum spacing between vehicles in same lane (strong collision prevention)
        vehicles_by_lane = {}
        for v in self.vehicles:
            if v.lane not in vehicles_by_lane:
                vehicles_by_lane[v.lane] = []
            vehicles_by_lane[v.lane].append(v)

        # Sort vehicles by position in each lane
        for lane in vehicles_by_lane:
            vehicles_by_lane[lane].sort(key=lambda x: x.position)

            # Enforce minimum spacing
            for i in range(len(vehicles_by_lane[lane]) - 1):
                v_rear = vehicles_by_lane[lane][i]
                v_front = vehicles_by_lane[lane][i + 1]

                min_gap = v_rear.length * 3.0  # Minimum 3x vehicle length spacing
                actual_gap = v_front.position - v_rear.position

                if actual_gap < min_gap:
                    # Push front vehicle forward
                    v_front.position = v_rear.position + min_gap
                    # Adjust speed to maintain gap
                    v_front.speed = max(v_front.speed, v_rear.speed + 1.0)

        # Check for collisions (for statistics only, spacing is already enforced above)
        for i, v1 in enumerate(self.vehicles):
            for v2 in self.vehicles[i + 1 :]:
                if v1.lane == v2.lane:
                    distance = abs(v1.position - v2.position)
                    if distance < v1.length * 2.0:
                        self.collision_count += 1

        # Remove vehicles beyond road
        self.vehicles = [
            v for v in self.vehicles if v.position <= self.road_length + 10.0
        ]

    def _spawn_vehicle(self):
        """Spawn a new vehicle at the start of the road."""
        lane = rd.randrange(self.num_lanes)
        pos = 0.0
        speed = np.clip(
            np.random.normal(self.max_speed * 0.85, 5.0),
            self.min_speed,
            self.max_speed,
        )

        # Only spawn if there's enough space
        can_spawn = True
        for v in self.vehicles:
            # Check both vehicles in lane and those transitioning to it
            is_in_lane = v.lane == lane or (
                v.is_changing_lane and v.target_lane == lane
            )
            if is_in_lane and v.position < 20.0:
                can_spawn = False
                break

        if can_spawn:
            self.vehicles.append(
                Vehicle(
                    lane=lane,
                    position=pos,
                    speed=speed,
                    max_speed=self.max_speed,
                    desired_speed=speed,
                    is_controlled=False,
                )
            )

    def _spawn_vehicle_controlled(self):
        """Spawn a new RL-controlled vehicle for multi-vehicle mode."""
        lane = rd.randrange(self.num_lanes)
        pos = 0.0
        speed = rd.uniform(self.min_speed, self.max_speed)
        desired_speed = rd.uniform(self.min_speed, self.max_speed)

        # Only spawn if there's enough space (check for vehicles in lane or transitioning to it)
        can_spawn = True
        for v in self.vehicles:
            is_in_lane = v.lane == lane or (
                v.is_changing_lane and v.target_lane == lane
            )
            if is_in_lane and v.position < 20.0:
                can_spawn = False
                break

        if can_spawn:
            self.vehicles.append(
                Vehicle(
                    lane=lane,
                    position=pos,
                    speed=speed,
                    max_speed=self.max_speed,
                    desired_speed=desired_speed,
                    is_controlled=True,  # RL-controlled in multi-vehicle mode
                )
            )

    def _vehicle_in_front(self, subject):
        """Get the nearest vehicle ahead in the same lane."""
        candidates = [
            v
            for v in self.vehicles
            if v.lane == subject.lane and v.position > subject.position
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda x: x.position)

    def _get_obs(self):
        """
        Build normalized observation based on control mode.

        Single-vehicle mode: Local view around ego vehicle (15 features)
        Multi-vehicle mode: Global state of all vehicles (max_vehicles * 5 + 6 features)
        """
        if self.multi_vehicle_control:
            # Multi-vehicle mode: global state observation
            obs = []

            # Per-vehicle features (padded to max_vehicles)
            for i in range(self.max_vehicles):
                if i < len(self.vehicles):
                    v = self.vehicles[i]
                    # Normalized features: [lane, position, speed, acceleration, desired_speed]
                    lane_norm = v.lane / max(1, self.num_lanes - 1)
                    pos_norm = np.clip(v.position / self.road_length, 0, 1)
                    speed_norm = np.clip(
                        (v.speed - self.min_speed) / (self.max_speed - self.min_speed),
                        0,
                        1,
                    )
                    acc_norm = np.clip(
                        (v.acceleration + 5) / 10, 0, 1
                    )  # [-5, 5] -> [0, 1]
                    desired_speed_norm = np.clip(
                        (v.desired_speed - self.min_speed)
                        / (self.max_speed - self.min_speed),
                        0,
                        1,
                    )

                    obs.extend(
                        [lane_norm, pos_norm, speed_norm, acc_norm, desired_speed_norm]
                    )
                else:
                    # Padding for non-existent vehicles
                    obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            # Global features (fixed size, independent of num_lanes)
            # These aggregate statistics work for any lane configuration

            # 1. Global average speed
            if len(self.vehicles) > 0:
                avg_speed_all = sum(v.speed for v in self.vehicles) / len(self.vehicles)
                avg_speed_norm = np.clip(
                    (avg_speed_all - self.min_speed)
                    / (self.max_speed - self.min_speed),
                    0,
                    1,
                )
            else:
                avg_speed_norm = 0.5

            # 2. Speed variance (smoothness indicator)
            if len(self.vehicles) > 1:
                speed_std = np.std([v.speed for v in self.vehicles])
                speed_std_norm = np.clip(
                    speed_std / 10.0, 0, 1
                )  # Normalize by typical std
            else:
                speed_std_norm = 0.0

            # 3. Number of vehicles (traffic density)
            num_vehicles_norm = len(self.vehicles) / self.max_vehicles

            # 4. Number of lanes (tells network the context)
            num_lanes_norm = self.num_lanes / 10.0  # Assume max 10 lanes typical

            # 5. Lane balance score (how evenly distributed)
            lane_counts = [
                sum(1 for v in self.vehicles if v.lane == i)
                for i in range(self.num_lanes)
            ]
            if len(self.vehicles) >= self.num_lanes:
                ideal_per_lane = len(self.vehicles) / self.num_lanes
                balance_variance = np.var(lane_counts) / max(1, ideal_per_lane)
                lane_balance_norm = np.clip(
                    1.0 - balance_variance / 5.0, 0, 1
                )  # 1.0 = perfect balance
            else:
                lane_balance_norm = 1.0

            # 6. Collision risk indicator
            collision_risk = min(self.collision_count / 10.0, 1.0)  # Normalize

            obs.extend(
                [
                    avg_speed_norm,
                    speed_std_norm,
                    num_vehicles_norm,
                    num_lanes_norm,
                    lane_balance_norm,
                    collision_risk,
                ]
            )

            return np.array(obs, dtype=np.float32)

        else:
            # Single-vehicle mode: local ego-centric observation
            if self.ego_index is None:
                raise RuntimeError(
                    "Environment must be reset before getting observations."
                )

            ego = self.vehicles[self.ego_index]

            # Ego state
            own_lane = ego.lane / max(1, self.num_lanes - 1)
            own_speed = np.clip(
                (ego.speed - self.min_speed) / (self.max_speed - self.min_speed), 0, 1
            )
            own_acc = np.clip(
                (ego.acceleration + 5) / 10, 0, 1
            )  # normalize from [-5, 5] to [0, 1]

            # Front vehicle in current lane
            gap_front, rel_speed_front = self._gap_and_rel_speed(
                ego, ego.lane, direction="front"
            )
            gap_front = np.clip(gap_front / 100.0, 0, 1)
            rel_speed_front = np.clip(
                (rel_speed_front + 20) / 40, 0, 1
            )  # normalize from [-20, 20] to [0, 1]

            # Left lane info
            if ego.lane > 0:
                gap_left_front, _ = self._gap_and_rel_speed(
                    ego, ego.lane - 1, direction="front"
                )
                gap_left_back, _ = self._gap_and_rel_speed(
                    ego, ego.lane - 1, direction="back"
                )
                gap_left_front = np.clip(gap_left_front / 100.0, 0, 1)
                gap_left_back = np.clip(gap_left_back / 100.0, 0, 1)
            else:
                gap_left_front = gap_left_back = 0.0

            # Right lane info
            if ego.lane < self.num_lanes - 1:
                gap_right_front, _ = self._gap_and_rel_speed(
                    ego, ego.lane + 1, direction="front"
                )
                gap_right_back, _ = self._gap_and_rel_speed(
                    ego, ego.lane + 1, direction="back"
                )
                gap_right_front = np.clip(gap_right_front / 100.0, 0, 1)
                gap_right_back = np.clip(gap_right_back / 100.0, 0, 1)
            else:
                gap_right_front = gap_right_back = 0.0

            # Lane densities and average speeds
            lane_densities = []
            lane_avg_speeds = []
            for lane_id in range(self.num_lanes):
                density = self._lane_density(lane_id) / 50.0  # normalize
                avg_speed = self._lane_avg_speed(lane_id)
                lane_densities.append(np.clip(density, 0, 1))
                lane_avg_speeds.append(np.clip(avg_speed, 0, 1))

            obs = np.array(
                [
                    own_lane,
                    own_speed,
                    own_acc,
                    gap_front,
                    rel_speed_front,
                    gap_left_front,
                    gap_left_back,
                    gap_right_front,
                    gap_right_back,
                    *lane_densities,
                    *lane_avg_speeds,
                ],
                dtype=np.float32,
            )

        # Ensure correct size
        if len(obs) < 15:
            obs = np.pad(obs, (0, 15 - len(obs)), "constant", constant_values=0.0)
        elif len(obs) > 15:
            obs = obs[:15]

        return obs

    def _gap_and_rel_speed(self, ego, lane, direction="front"):
        """Get gap distance and relative speed to nearest vehicle in specified direction."""
        if direction == "front":
            candidates = [
                v for v in self.vehicles if v.lane == lane and v.position > ego.position
            ]
        else:  # back
            candidates = [
                v for v in self.vehicles if v.lane == lane and v.position < ego.position
            ]

        if not candidates:
            return 100.0, 0.0

        if direction == "front":
            nearest = min(candidates, key=lambda x: x.position)
            gap = nearest.position - ego.position
        else:
            nearest = max(candidates, key=lambda x: x.position)
            gap = ego.position - nearest.position

        rel_speed = ego.speed - nearest.speed
        return gap, rel_speed

    def _lane_density(self, lane):
        """Count vehicles in a lane."""
        count = sum(1 for v in self.vehicles if v.lane == lane)
        return count

    def _lane_avg_speed(self, lane):
        """Get normalized average speed in a lane."""
        vehicles_in_lane = [v for v in self.vehicles if v.lane == lane]
        if not vehicles_in_lane:
            return 1.0  # empty lane - assume max speed

        avg_speed = sum(v.speed for v in vehicles_in_lane) / len(vehicles_in_lane)
        return (avg_speed - self.min_speed) / (self.max_speed - self.min_speed)

    def _compute_reward(self):
        """
        Dynamic, context-aware reward function for scalable traffic optimization.

        Adapts to:
        - Different lane counts (3, 5, 10, 20 lanes)
        - Variable traffic density
        - Diverse speed distributions
        - Lane blockages/construction scenarios

        Core philosophy:
        1. Compute dynamic speed tiers (percentiles, not fixed thresholds)
        2. Evaluate lane health (density, flow, variance)
        3. Score vehicle-lane fit quality
        4. Optimize global throughput with balanced utilization
        5. Minimize unnecessary disruptions (lane changes)
        """
        reward = 0.0

        if self.multi_vehicle_control:
            # Multi-vehicle mode: Global traffic optimization
            if len(self.vehicles) == 0:
                return 0.0

            # PHASE 1: Compute dynamic speed distribution
            all_desired_speeds = [v.desired_speed for v in self.vehicles]
            if len(all_desired_speeds) < 3:
                # Too few vehicles, use fixed thresholds as fallback
                slow_threshold = self.max_speed * 0.33
                fast_threshold = self.max_speed * 0.67
            else:
                # Dynamic percentiles adapt to current traffic
                slow_threshold = np.percentile(all_desired_speeds, 33)
                fast_threshold = np.percentile(all_desired_speeds, 67)

            # PHASE 2: Compute lane health metrics
            lane_metrics = {}
            for lane_id in range(self.num_lanes):
                vehicles_in_lane = [v for v in self.vehicles if v.lane == lane_id]

                if len(vehicles_in_lane) > 0:
                    avg_speed = np.mean([v.speed for v in vehicles_in_lane])
                    avg_desired_speed = np.mean(
                        [v.desired_speed for v in vehicles_in_lane]
                    )
                    speed_variance = (
                        np.var([v.speed for v in vehicles_in_lane])
                        if len(vehicles_in_lane) > 1
                        else 0.0
                    )
                    density = len(vehicles_in_lane) / (
                        self.road_length / 100.0
                    )  # vehicles per 100m
                    lane_throughput = len(vehicles_in_lane) * avg_speed
                else:
                    avg_speed = self.max_speed
                    avg_desired_speed = self.max_speed
                    speed_variance = 0.0
                    density = 0.0
                    lane_throughput = 0.0

                lane_metrics[lane_id] = {
                    "count": len(vehicles_in_lane),
                    "avg_speed": avg_speed,
                    "avg_desired_speed": avg_desired_speed,
                    "speed_variance": speed_variance,
                    "density": density,
                    "throughput": lane_throughput,
                }

            # PHASE 3: Dynamic lane allocation scoring
            placement_score = 0.0

            for v in self.vehicles:
                vehicle_lane = v.target_lane if v.is_changing_lane else v.lane

                # Calculate ideal lane position based on speed percentile
                speed_percentile = np.searchsorted(
                    sorted(all_desired_speeds), v.desired_speed
                ) / len(all_desired_speeds)
                # Fast vehicles (high percentile) → left (low lane index)
                # Slow vehicles (low percentile) → right (high lane index)
                ideal_relative_position = 1.0 - speed_percentile
                ideal_lane = ideal_relative_position * (self.num_lanes - 1)

                # Current lane as relative position
                current_relative_position = vehicle_lane / max(1, self.num_lanes - 1)

                # Distance from ideal position
                position_error = abs(ideal_lane - vehicle_lane)
                position_match_score = max(0, 1.0 - position_error / self.num_lanes)
                placement_score += position_match_score

                # Bonus for being in optimal tier
                if v.desired_speed < slow_threshold:  # Slow tier
                    if vehicle_lane >= self.num_lanes // 2:  # Right half
                        placement_score += 0.5
                elif v.desired_speed > fast_threshold:  # Fast tier
                    if vehicle_lane < (self.num_lanes + 1) // 2:  # Left half
                        placement_score += 0.5
                else:  # Medium tier
                    middle_lane = self.num_lanes // 2
                    if abs(vehicle_lane - middle_lane) <= 1:  # Near middle
                        placement_score += 0.5

                # Penalty for being in overcrowded lanes (dynamic capacity awareness)
                lane_density = lane_metrics[vehicle_lane]["density"]
                optimal_density = 0.12  # 12 vehicles per 100m ≈ 8.3m spacing
                if lane_density > optimal_density * 1.5:  # 50% over capacity
                    overcrowding_penalty = (lane_density - optimal_density) * 2.0
                    placement_score -= min(overcrowding_penalty, 2.0)

                # Penalty for blocking (slower vehicle ahead of faster vehicle)
                for other in self.vehicles:
                    if other.lane == vehicle_lane and other.position > v.position:
                        distance = other.position - v.position
                        if distance < 50.0:  # Within interaction range
                            if v.desired_speed < other.desired_speed - 3.0:
                                # This vehicle is slower and potentially blocking
                                blocking_severity = (
                                    other.desired_speed - v.desired_speed
                                ) / 10.0
                                placement_score -= min(blocking_severity, 1.5)

            # Normalize by vehicle count
            placement_score /= max(1, len(self.vehicles))
            reward += 12.0 * placement_score  # Primary objective

            # PHASE 4: Global throughput optimization
            total_throughput = sum(v.speed for v in self.vehicles)
            max_possible_throughput = len(self.vehicles) * self.max_speed
            throughput_ratio = total_throughput / max(max_possible_throughput, 1.0)
            reward += 5.0 * throughput_ratio  # High throughput = efficient system

            # PHASE 5: Lane balance - prevent all vehicles clustering in one lane
            lane_counts = [lane_metrics[i]["count"] for i in range(self.num_lanes)]
            if (
                len(self.vehicles) > 1
            ):  # Need at least 2 vehicles to have balance issues
                # Calculate how evenly distributed vehicles are
                ideal_per_lane = len(self.vehicles) / self.num_lanes
                balance_penalty = sum(
                    [abs(count - ideal_per_lane) for count in lane_counts]
                ) / len(self.vehicles)
                reward -= 2.0 * balance_penalty  # Penalize severe imbalance

            # PHASE 6: Per-lane flow quality
            per_lane_flow_score = 0.0
            for lane_id in range(self.num_lanes):
                metrics = lane_metrics[lane_id]
                if metrics["count"] > 1:
                    # Reward smooth flow within each lane (low variance)
                    smoothness = 1.0 / (1.0 + metrics["speed_variance"] / 10.0)
                    per_lane_flow_score += smoothness * metrics["count"]

            if len(self.vehicles) > 0:
                per_lane_flow_score /= len(self.vehicles)
                reward += 3.0 * per_lane_flow_score

            # PHASE 7: Stability - penalize lane changes (prevents oscillation)
            if self.lane_changes_this_step > 0:
                # STRONG penalty to prevent oscillation
                # Each lane change is very costly - stability is critical
                traffic_density = len(self.vehicles) / self.max_vehicles
                # Base penalty 5.0, scales up to 8.0 in heavy traffic
                change_penalty = 5.0 + (3.0 * traffic_density)
                reward -= change_penalty * self.lane_changes_this_step

            # PHASE 8: Safety - critical for all scenarios
            safety_penalty = 0.0
            for v in self.vehicles:
                front = self._vehicle_in_front(v)
                if front and front.lane == v.lane:
                    gap = front.position - v.position - v.length
                    if gap < 3.0:
                        safety_penalty += 2.0  # Collision imminent
                    elif gap < 8.0:
                        safety_penalty += 0.5  # Too close
            reward -= safety_penalty

            # PHASE 9: Collision penalty (absolute constraint)
            if self.collision_count > 0:
                reward -= 10.0

        else:
            # Single-vehicle mode: ego-centric optimization
            if self.ego_index is None:
                raise RuntimeError("Environment must be reset before computing reward.")

            ego = self.vehicles[self.ego_index]

            # 1. Speed reward: encourage driving at desired speed
            speed_ratio = ego.speed / ego.desired_speed
            reward += 2.0 * speed_ratio  # main reward component

            # 2. Safety: penalize unsafe following distances
            front = self._vehicle_in_front(ego)
            if front:
                gap = front.position - ego.position - ego.length
                if gap < 3.0:
                    reward -= 5.0  # severe penalty for collision risk
                elif gap < 8.0:
                    reward -= 1.0  # moderate penalty
                elif gap > 50.0:
                    # Slight penalty for being too far - could merge
                    reward -= 0.1

            # 3. Smoothness: penalize hard braking (stop-and-go indicator)
            if ego.acceleration < -ego.comfortable_deceleration * 0.5:
                reward -= 0.5

            # 4. Global flow: reward good overall traffic flow
            if len(self.vehicles) > 1:
                avg_speed = sum(v.speed for v in self.vehicles) / len(self.vehicles)
                flow_ratio = avg_speed / self.max_speed
                reward += 0.5 * flow_ratio

                # Penalize speed variance (stop-and-go detection)
                speed_variance = np.var([v.speed for v in self.vehicles])
                reward -= 0.01 * speed_variance

            # 5. Lane efficiency: slight bonus for being in faster lanes when possible
            current_lane_speed = self._lane_avg_speed(ego.lane)
            reward += 0.2 * current_lane_speed

            # 6. Collision penalty
            if self.collision_count > 0:
                reward -= 10.0

        return reward

    def get_metrics(self):
        """Return current episode metrics for logging."""
        if not self.speed_history:
            return {}

        return {
            "avg_speed": np.mean(self.speed_history),
            "speed_std": np.std(self.speed_history),
            "lane_changes": self.lane_change_count,
            "hard_braking_events": self.hard_braking_count,
            "collisions": self.collision_count,
            "total_reward": self.total_reward,
            "num_vehicles": len(self.vehicles),
            "step": self.step_count,
            "max_steps": self.max_episode_steps,
        }

    def render(self):
        """Render the environment using OpenCV visualization."""
        if self.render_mode == "human":
            if self.renderer is None:
                from .renderer import HighwayRenderer

                self.renderer = HighwayRenderer(self.render_config)

            # Get current observation for overlay
            obs = (
                self._get_obs()
                if (self.ego_index is not None or self.multi_vehicle_control)
                else None
            )

            # Get current stats
            stats = self.get_metrics()

            # Render frame
            frame = self.renderer.render(
                vehicles=self.vehicles,
                ego_index=self.ego_index,
                num_lanes=self.num_lanes,
                road_length=self.road_length,
                current_action=self.last_action,
                observation=obs,
                stats=stats,
                multi_vehicle_control=self.multi_vehicle_control,
            )

            # Show and handle keyboard input
            key = self.renderer.show()
            return frame
        else:
            # Text rendering fallback
            print(
                f"Step {self.step_count}, Vehicles: {len(self.vehicles)}, Avg Speed: {np.mean(self.speed_history[-10:]) if self.speed_history else 0:.1f} m/s"
            )
            return None

    def close(self):
        """Close the environment and renderer."""
        if self.renderer is not None:
            self.renderer.close()
        super().close()
