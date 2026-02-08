"""
OpenCV-based renderer for highway environment visualization.
Provides modern, clean UI with real-time metrics and visual feedback.
"""

from collections import deque
from typing import Any, cast

import cv2
import numpy as np

from shared_src.common import Config


class HighwayRenderer:
    """
    Professional visualization renderer for highway traffic simulation.
    Creates a modern, clean interface showing lanes, vehicles, and metrics.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the renderer with configuration.

        Args:
            config: Visualization configuration dictionary
        """
        self.config = config
        self.window_name = (
            cast(str, Config.get("project_name", "Lane Allocation"))
            + " - Visualization"
        )

        # Window dimensions
        self.window_width = config.get("window_width", 1400)
        self.window_height = config.get("window_height", 900)

        # Colors (BGR format)
        colors = config.get("colors", {})
        self.color_bg = tuple(colors.get("background", [30, 30, 30]))
        self.color_road = tuple(colors.get("road", [50, 50, 50]))
        self.color_lane_line = tuple(colors.get("lane_line", [200, 200, 200]))
        self.color_lane_edge = tuple(colors.get("lane_edge", [255, 255, 255]))
        self.color_ego = tuple(colors.get("ego_vehicle", [0, 255, 0]))
        self.color_other = tuple(colors.get("other_vehicle", [100, 150, 255]))
        self.color_fast = tuple(colors.get("fast_vehicle", [0, 255, 0]))
        self.color_slow = tuple(colors.get("slow_vehicle", [0, 0, 255]))
        self.color_text = tuple(colors.get("text", [255, 255, 255]))
        self.color_text_bg = tuple(colors.get("text_bg", [0, 0, 0]))
        self.color_panel_bg = tuple(colors.get("panel_bg", [40, 40, 40]))
        self.color_success = tuple(colors.get("success", [0, 255, 0]))
        self.color_warning = tuple(colors.get("warning", [0, 165, 255]))
        self.color_danger = tuple(colors.get("danger", [0, 0, 255]))

        # Layout
        self.highway_height_ratio = config.get("highway_height_ratio", 0.75)
        self.metrics_panel_width = config.get("metrics_panel_width", 300)

        # Highway area dimensions
        self.highway_width = self.window_width - self.metrics_panel_width
        self.highway_height = int(self.window_height * self.highway_height_ratio)

        # Vehicle rendering
        vehicle_config = config.get("vehicle", {})
        self.vehicle_width = vehicle_config.get("width", 25)
        self.vehicle_length = vehicle_config.get("length", 50)
        self.vehicle_corner_radius = vehicle_config.get("corner_radius", 5)
        self.vehicle_border_width = vehicle_config.get("border_width", 2)

        # Speed visualization
        speed_config = config.get("speed", {})
        self.show_speed_indicators = speed_config.get("show_indicators", True)
        self.show_speed_gradient = speed_config.get("show_gradient", True)
        self.speed_font_scale = speed_config.get("font_scale", 0.4)

        # Action indicators
        action_config = config.get("action", {})
        self.show_action_arrows = action_config.get("show_arrows", True)
        self.arrow_size = action_config.get("arrow_size", 20)
        self.arrow_offset = action_config.get("arrow_offset", 30)

        # Metrics
        metrics_config = config.get("metrics", {})
        self.show_metrics_panel = metrics_config.get("show_panel", True)
        self.show_graphs = metrics_config.get("show_graphs", True)
        self.graph_history = metrics_config.get("graph_history", 100)
        self.metrics_font_scale = metrics_config.get("font_scale", 0.5)
        self.metrics_line_spacing = metrics_config.get("line_spacing", 25)

        # Observation overlay
        obs_config = config.get("observation", {})
        self.show_observation = obs_config.get("show_overlay", False)
        self.obs_opacity = obs_config.get("opacity", 0.7)

        # Initialize frame
        self.frame = None

        # Metrics history for graphs
        self.speed_history = deque(maxlen=self.graph_history)
        self.reward_history = deque(maxlen=self.graph_history)

        # Current stats
        self.current_stats = {}

        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)

    def render(
        self,
        vehicles: list,
        ego_index: int | None,
        num_lanes: int,
        road_length: float,
        current_action: int | np.ndarray = 0,
        observation: np.ndarray | None = None,
        stats: dict[str, Any] | None = None,
        multi_vehicle_control: bool = False,
    ) -> np.ndarray:
        """
        Render the current state of the highway environment.

        Args:
            vehicles: List of Vehicle objects
            ego_index: Index of ego vehicle in vehicles list (single-vehicle mode)
            num_lanes: Number of lanes
            road_length: Total road length in meters
            current_action: Current action(s) - int for single, array for multi-vehicle
            observation: Current observation array
            stats: Dictionary of current statistics
            multi_vehicle_control: Whether in centralized multi-vehicle mode

        Returns:
            RGB frame as numpy array
        """
        # Create blank frame
        self.frame = np.zeros(
            (self.window_height, self.window_width, 3), dtype=np.uint8
        )
        self.frame[:] = self.color_bg

        # Update stats
        if stats:
            self.current_stats = stats
            if "avg_speed" in stats:
                self.speed_history.append(stats["avg_speed"])
            if "reward" in stats:
                self.reward_history.append(stats["reward"])

        # Draw highway
        self._draw_highway(num_lanes, multi_vehicle_control)

        # Draw vehicles
        self._draw_vehicles(
            vehicles,
            ego_index,
            num_lanes,
            road_length,
            current_action,
            multi_vehicle_control,
        )

        # Draw metrics panel
        if self.show_metrics_panel:
            self._draw_metrics_panel()

        # Draw observation overlay if enabled
        if self.show_observation and observation is not None:
            self._draw_observation_overlay(observation)

        # Draw controls hint
        self._draw_controls_hint()

        return self.frame

    def _draw_highway(self, num_lanes: int, multi_vehicle_control: bool = False):
        """Draw the highway with lanes."""
        if self.frame is None:
            return

        # Highway background
        cv2.rectangle(
            self.frame,
            (0, 0),
            (self.highway_width, self.highway_height),
            self.color_road,
            -1,
        )

        # Lane width
        lane_width = self.highway_width // num_lanes

        # Draw lane dividers
        for i in range(1, num_lanes):
            x = i * lane_width
            # Dashed line effect
            for y in range(0, self.highway_height, 40):
                cv2.line(
                    self.frame,
                    (x, y),
                    (x, min(y + 20, self.highway_height)),
                    self.color_lane_line,
                    2,
                )

        # Draw road edges
        cv2.line(
            self.frame,
            (0, 0),
            (0, self.highway_height),
            self.color_lane_edge,
            3,
        )
        cv2.line(
            self.frame,
            (self.highway_width - 1, 0),
            (self.highway_width - 1, self.highway_height),
            self.color_lane_edge,
            3,
        )

        # Draw lane labels at the top
        for i in range(num_lanes):
            x = i * lane_width + lane_width // 2
            label = f"Lane {i}"
            self._draw_text_with_background(
                label,
                (x - 30, 20),
                font_scale=0.5,
                color=self.color_text,
                thickness=1,
            )

        # Mode indicator banner
        if multi_vehicle_control:
            mode_text = "CENTRALIZED TRAFFIC CONTROL - ALL VEHICLES RL-CONTROLLED"
            color_key = "RED=Slow | ORANGE=Medium | GREEN=Fast"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2

            # Main mode text
            text_size = cv2.getTextSize(mode_text, font, font_scale, thickness)[0]
            text_x = (self.highway_width - text_size[0]) // 2
            text_y = 60

            # Background rectangle
            cv2.rectangle(
                self.frame,
                (text_x - 10, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 10, text_y + 5),
                (0, 120, 200),  # Orange background
                -1,
            )

            cv2.putText(
                self.frame,
                mode_text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

            # Color key text
            key_size = cv2.getTextSize(color_key, font, 0.4, 1)[0]
            key_x = (self.highway_width - key_size[0]) // 2
            key_y = 85

            cv2.putText(
                self.frame,
                color_key,
                (key_x, key_y),
                font,
                0.4,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

    def _draw_vehicles(
        self,
        vehicles: list,
        ego_index: int | None,
        num_lanes: int,
        road_length: float,
        current_action: int | np.ndarray,
        multi_vehicle_control: bool = False,
    ):
        """
        Draw all vehicles on the highway.

        Args:
            vehicles: List of Vehicle objects
            ego_index: Index of ego vehicle (single-vehicle mode only)
            num_lanes: Number of lanes
            road_length: Total road length in meters
            current_action: Single action (int) or array of actions (multi-vehicle)
            multi_vehicle_control: Whether in multi-vehicle mode
        """
        lane_width = self.highway_width // num_lanes

        for i, vehicle in enumerate(vehicles):
            is_ego = i == ego_index and not multi_vehicle_control

            # Calculate position on screen
            # Map position (0 to road_length) to screen y (highway_height to 0)
            # Vehicle at position 0 is at bottom, position road_length is at top
            progress = vehicle.position / road_length
            y = int(self.highway_height * (1 - progress))

            # Lane position with smooth transition animation
            if vehicle.is_changing_lane:
                # Interpolate between source and target lane
                visual_lane = (
                    vehicle.source_lane
                    + (vehicle.target_lane - vehicle.source_lane)
                    * vehicle.lane_transition_progress
                )
            else:
                visual_lane = float(vehicle.lane)

            x = int(visual_lane * lane_width + lane_width // 2)

            # Determine color
            if multi_vehicle_control:
                # Multi-vehicle mode: color by desired speed tier
                speed_ratio = vehicle.desired_speed / vehicle.max_speed
                if speed_ratio < 0.4:
                    color = self.color_slow  # Slow vehicles (should be right lane)
                elif speed_ratio < 0.7:
                    color = self.color_other  # Medium speed
                else:
                    color = self.color_fast  # Fast vehicles (should be left lane)
            elif is_ego:
                color = self.color_ego
            elif self.show_speed_gradient:
                # Color by speed (gradient from red to green)
                speed_ratio = vehicle.speed / vehicle.max_speed
                color = self._speed_to_color(speed_ratio)
            else:
                color = self.color_other

            # Draw vehicle as rounded rectangle
            self._draw_rounded_rectangle(
                x - self.vehicle_width // 2,
                y - self.vehicle_length // 2,
                self.vehicle_width,
                self.vehicle_length,
                color,
                is_ego
                or multi_vehicle_control,  # Border for ego or all vehicles in multi-mode
            )

            # Draw speed indicator
            if self.show_speed_indicators:
                speed_kmh = vehicle.speed * 3.6  # m/s to km/h
                speed_text = f"{speed_kmh:.0f}"
                self._draw_text_with_background(
                    speed_text,
                    (x - 15, y),
                    font_scale=self.speed_font_scale,
                    color=(255, 255, 255),
                    thickness=1,
                )

            # Draw action arrows
            if self.show_action_arrows:
                if multi_vehicle_control:
                    # Multi-vehicle: show actions for all vehicles
                    if isinstance(current_action, np.ndarray) and i < len(
                        current_action
                    ):
                        action = current_action[i]
                        if action != 0:  # Only draw if not "keep lane"
                            self._draw_action_arrow(x, y, action)
                elif is_ego and current_action != 0:
                    # Single-vehicle: show action only for ego
                    assert isinstance(
                        current_action, int
                    ), "Expected single action for single-vehicle mode"
                    self._draw_action_arrow(x, y, current_action)

    def _draw_rounded_rectangle(
        self, x: int, y: int, width: int, height: int, color: tuple, is_ego: bool
    ):
        """Draw a rounded rectangle for a vehicle."""
        if self.frame is None:
            return

        # Simple rounded corners using circles
        radius = self.vehicle_corner_radius

        # Main rectangle
        cv2.rectangle(
            self.frame,
            (x + radius, y),
            (x + width - radius, y + height),
            color,
            -1,
        )
        cv2.rectangle(
            self.frame,
            (x, y + radius),
            (x + width, y + height - radius),
            color,
            -1,
        )

        # Corners
        cv2.circle(self.frame, (x + radius, y + radius), radius, color, -1)
        cv2.circle(self.frame, (x + width - radius, y + radius), radius, color, -1)
        cv2.circle(self.frame, (x + radius, y + height - radius), radius, color, -1)
        cv2.circle(
            self.frame,
            (x + width - radius, y + height - radius),
            radius,
            color,
            -1,
        )

        # Draw border for ego vehicle
        if is_ego:
            border_color = (255, 255, 255)  # White border
            cv2.rectangle(
                self.frame,
                (x, y),
                (x + width, y + height),
                border_color,
                self.vehicle_border_width,
            )

    def _draw_action_arrow(self, x: int, y: int, action: int):
        """Draw an arrow indicating the action."""
        if self.frame is None:
            return

        arrow_y = y - self.arrow_offset
        arrow_length = self.arrow_size

        if action == 1:  # Left
            start = (x + arrow_length // 2, arrow_y)
            end = (x - arrow_length // 2, arrow_y)
        else:  # Right
            start = (x - arrow_length // 2, arrow_y)
            end = (x + arrow_length // 2, arrow_y)

        cv2.arrowedLine(
            self.frame,
            start,
            end,
            (255, 255, 0),  # Yellow
            3,
            tipLength=0.4,
        )

    def _speed_to_color(self, speed_ratio: float) -> tuple:
        """Convert speed ratio to color (red=slow, green=fast)."""
        # Clamp between 0 and 1
        speed_ratio = max(0.0, min(1.0, speed_ratio))

        # Interpolate between red and green
        r = int(255 * (1 - speed_ratio))
        g = int(255 * speed_ratio)
        b = 0

        return (b, g, r)  # BGR format

    def _draw_metrics_panel(self):
        """Draw the metrics panel on the right side."""
        if self.frame is None:
            return

        panel_x = self.highway_width
        panel_width = self.metrics_panel_width

        # Panel background
        cv2.rectangle(
            self.frame,
            (panel_x, 0),
            (self.window_width, self.window_height),
            self.color_panel_bg,
            -1,
        )

        # Title
        y_offset = 30
        self._draw_text_with_background(
            "METRICS",
            (panel_x + 10, y_offset),
            font_scale=0.7,
            color=self.color_text,
            thickness=2,
        )

        y_offset += 40

        # Display stats
        stats = self.current_stats
        spacing = self.metrics_line_spacing

        # Average speed
        avg_speed = stats.get("avg_speed", 0)
        avg_speed_kmh = avg_speed * 3.6
        speed_color = (
            self.color_success
            if avg_speed_kmh > 100
            else self.color_warning if avg_speed_kmh > 80 else self.color_danger
        )
        self._draw_metric_row(
            "Avg Speed:",
            f"{avg_speed_kmh:.1f} km/h",
            panel_x + 10,
            y_offset,
            speed_color,
        )
        y_offset += spacing

        # Speed std
        speed_std = stats.get("speed_std", 0)
        std_color = (
            self.color_success
            if speed_std < 3
            else self.color_warning if speed_std < 6 else self.color_danger
        )
        self._draw_metric_row(
            "Speed Std:",
            f"{speed_std:.2f} m/s",
            panel_x + 10,
            y_offset,
            std_color,
        )
        y_offset += spacing

        # Lane changes
        lane_changes = stats.get("lane_changes", 0)
        self._draw_metric_row(
            "Lane Changes:",
            f"{lane_changes}",
            panel_x + 10,
            y_offset,
            self.color_text,
        )
        y_offset += spacing

        # Hard braking
        hard_braking = stats.get("hard_braking_events", 0)
        brake_color = (
            self.color_success
            if hard_braking < 5
            else self.color_warning if hard_braking < 15 else self.color_danger
        )
        self._draw_metric_row(
            "Hard Braking:",
            f"{hard_braking}",
            panel_x + 10,
            y_offset,
            brake_color,
        )
        y_offset += spacing

        # Collisions
        collisions = stats.get("collisions", 0)
        collision_color = self.color_success if collisions == 0 else self.color_danger
        self._draw_metric_row(
            "Collisions:",
            f"{collisions}",
            panel_x + 10,
            y_offset,
            collision_color,
        )
        y_offset += spacing

        # Total reward
        total_reward = stats.get("total_reward", 0)
        self._draw_metric_row(
            "Total Reward:",
            f"{total_reward:.1f}",
            panel_x + 10,
            y_offset,
            self.color_text,
        )
        y_offset += spacing

        # Number of vehicles
        num_vehicles = stats.get("num_vehicles", 0)
        self._draw_metric_row(
            "Vehicles:",
            f"{num_vehicles}",
            panel_x + 10,
            y_offset,
            self.color_text,
        )
        y_offset += spacing

        # Episode step
        step = stats.get("step", 0)
        max_steps = stats.get("max_steps", 300)
        self._draw_metric_row(
            "Step:",
            f"{step}/{max_steps}",
            panel_x + 10,
            y_offset,
            self.color_text,
        )
        y_offset += 40

        # Draw graphs if enabled
        if self.show_graphs:
            self._draw_graph(
                "Speed History (m/s)",
                self.speed_history,
                panel_x + 10,
                y_offset,
                panel_width - 20,
                100,
                max_val=35.0,
            )
            y_offset += 120

            if self.reward_history:
                self._draw_graph(
                    "Reward History",
                    self.reward_history,
                    panel_x + 10,
                    y_offset,
                    panel_width - 20,
                    100,
                )

    def _draw_metric_row(
        self,
        label: str,
        value: str,
        x: int,
        y: int,
        value_color: tuple,
    ):
        """Draw a metric row with label and colored value."""
        if self.frame is None:
            return

        # Draw label
        cv2.putText(
            self.frame,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.metrics_font_scale,
            self.color_text,
            1,
            cv2.LINE_AA,
        )

        # Draw value
        label_width = len(label) * 10
        cv2.putText(
            self.frame,
            value,
            (x + label_width + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.metrics_font_scale,
            value_color,
            1,
            cv2.LINE_AA,
        )

    def _draw_graph(
        self,
        title: str,
        data: deque,
        x: int,
        y: int,
        width: int,
        height: int,
        max_val: float | None = None,
    ):
        """Draw a line graph."""
        if len(data) < 2 or self.frame is None:
            return

        # Graph background
        cv2.rectangle(
            self.frame,
            (x, y),
            (x + width, y + height),
            (20, 20, 20),
            -1,
        )

        # Title
        cv2.putText(
            self.frame,
            title,
            (x + 5, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.color_text,
            1,
            cv2.LINE_AA,
        )

        # Convert data to numpy array
        data_array = np.array(list(data))

        if max_val is None:
            max_val = np.max(data_array) if len(data_array) > 0 else 1.0
        min_val = np.min(data_array) if len(data_array) > 0 else 0.0

        if max_val == min_val:
            max_val = min_val + 1

        # Normalize data
        normalized = (data_array - min_val) / (max_val - min_val)

        # Draw line
        points = []
        for i, val in enumerate(normalized):
            px = x + int((i / len(data)) * width)
            py = y + height - int(val * (height - 20))
            points.append((px, py))

        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(
                    self.frame,
                    points[i],
                    points[i + 1],
                    self.color_success,
                    2,
                )

        # Draw min/max values
        cv2.putText(
            self.frame,
            f"{max_val:.1f}",
            (x + width - 35, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            self.color_text,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            self.frame,
            f"{min_val:.1f}",
            (x + width - 35, y + height - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            self.color_text,
            1,
            cv2.LINE_AA,
        )

    def _draw_observation_overlay(self, observation: np.ndarray):
        """Draw observation values as an overlay."""
        if self.frame is None:
            return

        # Semi-transparent panel
        overlay = self.frame.copy()
        panel_height = 200
        panel_width = 300

        cv2.rectangle(
            overlay,
            (10, self.highway_height - panel_height - 10),
            (10 + panel_width, self.highway_height - 10),
            self.color_panel_bg,
            -1,
        )

        # Blend with main frame
        cv2.addWeighted(
            overlay,
            self.obs_opacity,
            self.frame,
            1 - self.obs_opacity,
            0,
            self.frame,
        )

        # Draw observation values
        obs_labels = [
            "Lane",
            "Speed",
            "Accel",
            "Gap Front",
            "Rel Speed Front",
            "Gap L-F",
            "Gap L-B",
            "Gap R-F",
            "Gap R-B",
            "Density L0",
            "Density L1",
            "Density L2",
            "Avg Speed L0",
            "Avg Speed L1",
            "Avg Speed L2",
        ]

        y_offset = self.highway_height - panel_height
        for i, (label, val) in enumerate(zip(obs_labels[:10], observation[:10])):
            text = f"{label}: {val:.2f}"
            cv2.putText(
                self.frame,
                text,
                (20, y_offset + 20 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.color_text,
                1,
                cv2.LINE_AA,
            )

    def _draw_controls_hint(self):
        """Draw control hints at the bottom."""
        hints = "Controls: [SPACE] Pause | [R] Reset | [Q] Quit | [+/-] Speed | [O] Toggle Obs"
        text_size = cv2.getTextSize(hints, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]

        x = (self.highway_width - text_size[0]) // 2
        y = self.window_height - 10

        self._draw_text_with_background(
            hints,
            (x, y),
            font_scale=0.4,
            color=self.color_text,
            thickness=1,
        )

    def _draw_text_with_background(
        self,
        text: str,
        position: tuple,
        font_scale: float,
        color: tuple,
        thickness: int,
    ):
        """Draw text with a background box."""
        if self.frame is None:
            return

        x, y = position

        # Get text size
        text_size = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0]

        # Draw background
        cv2.rectangle(
            self.frame,
            (x - 2, y - text_size[1] - 2),
            (x + text_size[0] + 2, y + 2),
            self.color_text_bg,
            -1,
        )

        # Draw text
        cv2.putText(
            self.frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def show(self) -> int:
        """
        Display the frame and handle keyboard input.

        Returns:
            Key code pressed (-1 if no key)
        """
        if self.frame is not None:
            cv2.imshow(self.window_name, self.frame)
            return cv2.waitKey(1)
        return -1

    def close(self):
        """Close the renderer window."""
        cv2.destroyWindow(self.window_name)

    def get_frame(self) -> np.ndarray | None:
        """Get the current frame."""
        return self.frame
