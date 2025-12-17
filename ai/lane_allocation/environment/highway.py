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
        self.ego_index = None
        self.multi_agent = config.get("multi_agent", False)  # control multiple vehicles

        # Validate configuration
        if self.max_speed <= self.min_speed:
            raise ValueError(
                f"max_speed ({self.max_speed}) must be greater than min_speed ({self.min_speed})"
            )
        if self.num_lanes < 1:
            raise ValueError(f"num_lanes must be >= 1, got {self.num_lanes}")
        if self.spawn_rate < 0:
            raise ValueError(f"spawn_rate must be >= 0, got {self.spawn_rate}")

        # action space: 0=keep_lane, 1=change_left, 2=change_right
        self.action_space = spaces.Discrete(3)

        # observation space: normalized features for ego + nearby vehicles
        # [own_lane, own_speed, own_acc, gap_front, rel_speed_front, gap_left_front, gap_left_back,
        #  gap_right_front, gap_right_back, lane_densities (3), avg_speed_per_lane (3)]
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
        self.hard_braking_count = 0
        self.collision_count = 0

        # Spawn initial traffic with varied speeds
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
        if self.ego_index is None:
            raise RuntimeError("Environment must be reset before stepping.")

        self.step_count += 1
        ego = self.vehicles[self.ego_index]
        old_lane = ego.lane

        # Apply ego action (lane change only)
        self._apply_action(ego, action)

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
        }

        return obs, reward, done, truncated, info

    def _apply_action(self, ego, action):
        """Apply lane change action. Actions: 0=keep, 1=left, 2=right"""
        if action == 1 and ego.lane > 0:
            # Check safety for left lane change
            if self._is_lane_change_safe(ego, ego.lane - 1):
                ego.lane -= 1
        elif action == 2 and ego.lane < self.num_lanes - 1:
            # Check safety for right lane change
            if self._is_lane_change_safe(ego, ego.lane + 1):
                ego.lane += 1
        # action 0: keep lane - do nothing

    def _is_lane_change_safe(self, ego, target_lane):
        """
        Enhanced safety check for lane changes.
        Checks both front and rear gaps in target lane.
        """
        min_front_gap = 15.0  # meters
        min_rear_gap = 10.0  # meters

        for v in self.vehicles:
            if v is ego or v.lane != target_lane:
                continue

            distance = v.position - ego.position

            # Vehicle ahead in target lane
            if distance > 0 and distance < min_front_gap:
                return False

            # Vehicle behind in target lane
            if distance < 0 and abs(distance) < min_rear_gap:
                # Also check if rear vehicle is approaching fast
                if v.speed > ego.speed + 5.0:  # 5 m/s = 18 km/h faster
                    return False
                return False

        return True

    def _update_vehicles_idm(self):
        """
        Update vehicle accelerations using Intelligent Driver Model (IDM).
        IDM creates realistic car-following behavior and stop-and-go dynamics.
        """
        for v in self.vehicles:
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

                # IDM desired gap
                s_star = (
                    v.min_spacing
                    + v.time_headway * v.speed
                    + (v.speed * speed_diff)
                    / (2 * np.sqrt(v.max_acceleration * v.comfortable_deceleration))
                )

                # IDM acceleration
                acc_free = v.max_acceleration * (1.0 - (v.speed / v.desired_speed) ** 4)
                acc_interaction = -v.max_acceleration * (s_star / max(gap, 0.1)) ** 2

                v.acceleration = acc_free + acc_interaction

            # Clamp acceleration
            v.acceleration = np.clip(
                v.acceleration,
                -v.comfortable_deceleration,
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

        # Check for collisions (vehicles too close)
        for i, v1 in enumerate(self.vehicles):
            for v2 in self.vehicles[i + 1 :]:
                if v1.lane == v2.lane:
                    distance = abs(v1.position - v2.position)
                    if distance < v1.length:
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
            if v.lane == lane and v.position < 20.0:
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
        Build normalized observation for ego vehicle.
        Features: lane position, speed, acceleration, gaps, relative speeds, lane densities.
        """
        if self.ego_index is None:
            raise RuntimeError("Environment must be reset before getting observations.")

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
        Comprehensive reward function optimizing for:
        1. High speed (efficiency)
        2. Safe following distance (no tailgating)
        3. Smooth traffic flow (no stop-and-go)
        4. Efficient lane usage
        5. Minimal unnecessary lane changes
        """
        if self.ego_index is None:
            raise RuntimeError("Environment must be reset before computing reward.")

        ego = self.vehicles[self.ego_index]
        reward = 0.0

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
        }

    def render(self):
        """Simple text rendering."""
        print(
            f"Step {self.step_count}, Vehicles: {len(self.vehicles)}, Avg Speed: {np.mean(self.speed_history[-10:]) if self.speed_history else 0:.1f} m/s"
        )
