from pydantic import BaseModel


class Vehicle(BaseModel):
    lane: int
    position: float
    speed: float
    max_speed: float
    length: float = 4.5  # meters
    acceleration: float = 0.0  # current acceleration m/s^2
    desired_speed: float = 30.0  # m/s - IDM parameter
    min_spacing: float = 2.0  # minimum gap to front vehicle (m)
    time_headway: float = 1.5  # desired time headway (s) - IDM parameter
    max_acceleration: float = 2.0  # m/s^2
    comfortable_deceleration: float = 3.0  # m/s^2
    is_controlled: bool = False  # True if controlled by RL agent
    lane_change_cooldown: float = 0.0  # seconds until next lane change allowed

    # Smooth lane change animation
    is_changing_lane: bool = False  # True if currently transitioning between lanes
    source_lane: int = 0  # Lane we're transitioning from
    target_lane: int = 0  # Lane we're transitioning to
    lane_transition_progress: float = 0.0  # 0.0 to 1.0
    lane_transition_duration: float = 1.5  # seconds for complete transition
