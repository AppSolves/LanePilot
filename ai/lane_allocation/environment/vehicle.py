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
