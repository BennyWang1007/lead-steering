import os
from dataclasses import dataclass
from typing import Optional, Union

import jaxtyping as jt
import numpy as np
import torch
from beartype import beartype

from lead.common.pid_controller import LateralPIDController, PIDController, get_throttle
from lead.expert.config_expert import ExpertConfig
from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.inference.open_loop_inference import OpenLoopInference, OpenLoopPrediction
from lead.tfv6.tfv6 import Prediction
from lead.training.config_training import TrainingConfig

from activation_steering.steering_manager.tfv6_steering_manager import Tfv6SteeringManager

np.set_printoptions(suppress=True)


@dataclass
@beartype
class ClosedLoopPrediction(OpenLoopPrediction):
    steer: float
    throttle: float
    brake: float
    waypoints_steer: float
    waypoints_throttle: float
    waypoints_brake: float
    route_steer: float
    target_speed_throttle: float
    target_speed_brake: float
    steering_alpha: Optional[Union[float, list[float]]] = None


class ClosedLoopInference(OpenLoopInference):
    @beartype
    def __init__(
        self,
        config_training: TrainingConfig,
        config_closed_loop: ClosedLoopConfig,
        config_expert: ExpertConfig,
        model_path: str,
        device: torch.device,
        prefix: str,
    ):
        super().__init__(
            config_training=config_training,
            config_open_loop=config_closed_loop,
            model_path=model_path,
            device=device,
            prefix=prefix,
        )
        self.config_expert = config_expert
        self.config_training = config_training
        self.config_closed_loop = config_closed_loop

        self.lateral_waypoint_controller = PIDController(
            k_p=self.config_closed_loop.turn_kp,
            k_i=self.config_closed_loop.turn_ki,
            k_d=self.config_closed_loop.turn_kd,
            n=self.config_closed_loop.turn_n,
        )
        self.longitudinal_waypoint_controller = PIDController(
            k_p=self.config_closed_loop.speed_kp,
            k_i=self.config_closed_loop.speed_ki,
            k_d=self.config_closed_loop.speed_kd,
            n=self.config_closed_loop.speed_n,
        )
        self.lateral_route_controller = LateralPIDController(self.config_closed_loop)
        self.longitudinal_target_speed_controller = PIDController(
            k_p=self.config_closed_loop.speed_kp,
            k_i=self.config_closed_loop.speed_ki,
            k_d=self.config_closed_loop.speed_kd,
            n=self.config_closed_loop.speed_n,
        )

        self.step = 4  # Constant so produced images start with 5, not really important
        self.inference_step = 0

        # Steering manager initialization
        steering_dir = os.environ.get("STEERING_VECTORS_DIR")
        feature_name = os.environ.get("STEERING_FEATURE_NAME", "decoder_layer5")
        self.steering_manager = (
            Tfv6SteeringManager(
                steering_root=steering_dir,
                device=self.device,
                feature_name=feature_name,
            )
            if steering_dir is not None
            else None
        )
        if self.steering_manager is None:
            print("[ClosedLoopInference.__init__] STEERING_VECTORS_DIR not set: "
                  "any steering alpha requested by a policy will be IGNORED.", flush=True)

        self.target_layer = 5
        if "decoder_layer" in feature_name:
            try:
                self.target_layer = int(feature_name.replace("decoder_layer", ""))
            except ValueError:
                self.target_layer = 5
        elif feature_name == "fused_features":
            self.target_layer = -1

        print(f"[ClosedLoopInference.__init__] Number of nets: {len(self.nets)}")
        for model_idx, net in enumerate(self.nets):
            print(f"    Net {model_idx}: {net.__class__.__name__}, Size: {sum(p.numel() for p in net.parameters())}")

        # if env var "DEBUG_LOG_STEERING_DIRECTION" is set, we will log the steering direction at each step
        self.debug_log_steering_direction = os.environ.get("DEBUG_LOG_STEERING_DIRECTION", "0") in ["1", "true", "True"]

    @staticmethod
    @beartype
    def _get_steering_direction_from_alphas(alphas: list[float] | None) -> int:
        """
        Determine steering direction based on the provided alphas.
        Returns:
            -1 for no steering, 0 for brake, 1 for left, 2 for right
        """
        if alphas is None or all(a <= 0.0 for a in alphas):
            return -1  # No steering

        if len(alphas) != 3:
            raise ValueError(f"Expected 3 alphas for left, straight, right, got {alphas}")

        # Order of actions is [brake, left, right]
        if alphas[0] > alphas[1] and alphas[0] > alphas[2]:
            return 0  # Brake
        elif alphas[1] > alphas[0] and alphas[1] > alphas[2]:
            return 1  # Left
        elif alphas[2] > alphas[0] and alphas[2] > alphas[1]:
            return 2  # Right
        else:
            return -1  # None

    # @staticmethod
    @beartype
    def _get_steering_direction(self, prediction: Prediction) -> tuple[int, float]:
        """
        Check future waypoints or route checkpoints for lateral sign
        Returns:
            0 for straight, 1 for left, 2 for right, -1 for no prediction
        """
        points = getattr(prediction, "pred_future_waypoints", None)

        if self.debug_log_steering_direction:
            print(f"[ClosedLoopInference.get_steering_direction] Prediction points: {points}")

        if points is None:
            points = getattr(prediction, "pred_route", None)
            if self.debug_log_steering_direction:
                print(f"[ClosedLoopInference.get_steering_direction] Route points: {points}")

        if points is not None and points.shape[1] > 0:
            mean_lat = points[0, :3, 0].mean().item()
            mean_lat_all = points[0, :, 0].mean().item()
            if self.debug_log_steering_direction:
                print(f"[ClosedLoopInference.get_steering_direction] Mean lateral offset(3/all): {mean_lat}/{mean_lat_all}")
            if abs(mean_lat) < 0.05:
                return 0, 0.0  # Straight
            # Using the mean lateral offset (x-axis) of the near-term trajectory
            # return torch.sign(points[0, :3, 0].mean()).item()
            if torch.sign(points[0, :3, 0].mean()).item() > 0:
                return 2, mean_lat  # Right
            else:
                return 1, mean_lat  # Left
        return -1, 0.0  # No prediction

    @staticmethod
    def _normalize_alpha(steering_alpha: Optional[Union[float, list[float]]]) -> Optional[list[float]]:
        """None -> None, scalar -> [a, a, a], 3-sequence -> list of 3 floats."""
        if steering_alpha is None:
            return None
        if isinstance(steering_alpha, (int, float, np.floating)):
            return [float(steering_alpha)] * 3
        if isinstance(steering_alpha, (list, tuple, np.ndarray)) and len(steering_alpha) == 3:
            return [float(a) for a in steering_alpha]
        raise ValueError(
            f"Steering alpha must be a float or a list of 3 floats, got {steering_alpha}",
        )

    def next_step(self):
        """Increment the inference step counter. Useful for external control of the step count."""
        self.inference_step += 1

    def predict(
        self,
        data: dict[str, torch.Tensor],
        steering_alpha: Optional[Union[float, list[float]]] = None,
    ) -> OpenLoopPrediction:
        """Run all ensemble nets and fuse them. NO controller (PID) state is touched.

        Safe to call several times per frame (e.g. one call per Depth-TTC candidate).
        """
        steering_alpha = self._normalize_alpha(steering_alpha)
        predictions: list[Prediction] = []
        for model_idx, net in enumerate(self.nets):
            steering_vector = None
            if self.steering_manager is not None:
                steering_vector = self.steering_manager.get_steering_vector(
                    model_idx=model_idx,
                    alpha_input=steering_alpha,
                )
            predictions.append(
                net.forward(
                    data=data,
                    steering_alpha=steering_alpha,
                    steering_vector=steering_vector,
                    target_layer=self.target_layer,
                ),
            )
        return super().ensemble(data, predictions)

    def control(
        self,
        data: dict[str, torch.Tensor],
        open_loop_prediction: OpenLoopPrediction,
        steering_alpha: Optional[Union[float, list[float]]] = None,
    ) -> ClosedLoopPrediction:
        """Turn the chosen open-loop prediction into steer/throttle/brake. Call ONCE per frame:
        the PID controllers are stateful (integral / derivative terms)."""
        print(f"[ClosedLoopInference.control] Inference step: {self.inference_step}")
        print(f"[ClosedLoopInference.control] Steering alpha: {steering_alpha}")
        return self._to_closed_loop(
            data, open_loop_prediction, self._normalize_alpha(steering_alpha))

    def forward(
        self,
        data: dict[str, torch.Tensor],
        steering_alpha: Optional[Union[float, list[float]]] = None,
    ) -> ClosedLoopPrediction:
        """Single-candidate convenience path: predict() followed by control()."""
        return self.control(data, self.predict(data, steering_alpha), steering_alpha)

    @jt.jaxtyped(typechecker=beartype)
    def execute_route_and_target_speed(
        self,
        pred_checkpoints: jt.Float[torch.Tensor, "1 num_checkpoints 2"],
        pred_target_speed: jt.Float[torch.Tensor, "1 1"],
        speed: jt.Float[torch.Tensor, "1 1"],
        ego_vehicle_location: float = 0.0,
        ego_vehicle_rotation: float = 0.0,
    ) -> tuple[float, float, float]:
        """Predicts vehicle control with a PID controller. Use route and target speed predictions.

        Args:
            pred_checkpoints: Predicted route checkpoints in ego-vehicle coordinates.
            pred_target_speed: Predicted target speed in m/s.
            speed: Current speed of the vehicle in m/s.
            ego_vehicle_location: Current lateral location of the ego vehicle.
            ego_vehicle_rotation: Current rotation of the ego vehicle.
        Returns:
            steer: Steering command in [-1, 1]
            throttle: Throttle command in [0, 1]
            brake: Brake command in [0, 1]
        """
        pred_checkpoints = pred_checkpoints[0].data.cpu().float().numpy()
        speed = float(speed[0].data.cpu().float().numpy())
        pred_target_speed = float(pred_target_speed[0].data.cpu().float().numpy())

        brake = bool(
            pred_target_speed < 0.01
            or (speed / pred_target_speed) > self.config_closed_loop.brake_ratio,
        )
        steer = self.lateral_route_controller.step(
            pred_checkpoints,
            speed,
            ego_vehicle_location,
            ego_vehicle_rotation,
            sensor_agent_steer_correction=self.config_closed_loop.sensor_agent_steer_correction,
        )
        throttle, brake = get_throttle(
            brake,
            pred_target_speed,
            speed,
            self.config_expert,
        )

        return steer, throttle, float(brake)

    @jt.jaxtyped(typechecker=beartype)
    def execute_waypoints(
        self,
        waypoints: jt.Float[torch.Tensor, "1 num_waypoints 2"],
        velocity: jt.Float[torch.Tensor, "1 1"],
    ) -> tuple[float, float, float]:
        """Predicts vehicle control with a PID controller. Use waypoint predictions.

        Args:
            waypoints: Predicted future waypoints in ego-vehicle coordinates.
            velocity: Current speed of the vehicle in m/s.
        Returns:
            steer: Steering command in [-1, 1]
            throttle: Throttle command in [0, 1]
            brake: Brake command in [0, 1]
        """
        waypoints = waypoints[0].data.cpu().float().numpy()
        speed = velocity[0].data.cpu().float().numpy()

        one_second = int(
            self.config_training.carla_fps // self.config_training.data_save_freq,
        )
        half_second = one_second // 2

        desired_speed = (
            np.linalg.norm(waypoints[half_second - 1] - waypoints[one_second - 1]) * 2.0
        )
        delta_speed = float(
            np.clip(
                desired_speed - speed,
                0.0,
                self.config_closed_loop.wp_delta_clip,
            )
        )

        brake = (desired_speed < self.config_closed_loop.brake_speed) or (
            (speed / desired_speed) > self.config_closed_loop.brake_ratio
        )
        throttle = self.longitudinal_waypoint_controller.step(delta_speed)
        throttle = throttle if not brake else 0.0

        if (
            self.config_closed_loop.tuned_aim_distance
        ):  # In LB2, we go faster, so we need to choose waypoints farther ahead
            # range [2.4, 10.5] same as in the disentangled rep.
            aim_distance = np.clip(0.975532 * speed + 1.915288, 24, 105) / 10
        else:
            # To replicate the slow TransFuser behaviour we have a different distance
            # inside and outside of intersections (detected by desired_speed)
            if desired_speed < self.config_closed_loop.aim_distance_threshold:
                aim_distance = self.config_closed_loop.aim_distance_slow
            else:
                aim_distance = self.config_closed_loop.aim_distance_fast

        # We follow the waypoint that is at least a certain distance away
        aim_index = waypoints.shape[0] - 1
        for index, predicted_waypoint in enumerate(waypoints):
            if np.linalg.norm(predicted_waypoint) >= aim_distance:
                aim_index = index
                break

        aim = waypoints[aim_index]
        angle = float(np.degrees(np.arctan2(aim[1], aim[0])) / 90.0)
        if speed < 0.01:
            # When we don't move we don't want the angle error to accumulate in the integral
            angle = 0.0
        if brake:
            angle = 0.0

        steer = self.lateral_waypoint_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)  # Valid steering values are in [-1,1]
        return float(steer), float(throttle), float(brake)

    @beartype
    def ensemble(
        self,
        data: dict[str, torch.Tensor],
        predictions: list[Prediction],
        steering_alpha: Optional[float | list[float]] = None,
    ) -> ClosedLoopPrediction:
        """
        Args:
            data: Dictionary containing the input data
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            ClosedLoopPrediction: Dataclass containing all predictions
        """
        open_loop_prediction: OpenLoopPrediction = super().ensemble(data, predictions)
        return self._to_closed_loop(data, open_loop_prediction, steering_alpha)

    def _to_closed_loop(
        self,
        data: dict[str, torch.Tensor],
        open_loop_prediction: OpenLoopPrediction,
        steering_alpha: Optional[Union[float, list[float]]] = None,
    ) -> ClosedLoopPrediction:
        print(f"[ClosedLoopInference._to_closed_loop] Inference step: {self.inference_step}")
        ego_speed = data["speed"].to(self.device, dtype=torch.float32).unsqueeze(1)

        # Convert high-level commands to vehicle controls
        steer = throttle = brake = waypoints_steer = waypoints_throttle = (
            waypoints_brake
        ) = route_steer = target_speed_throttle = target_speed_brake = None

        if open_loop_prediction.pred_route is not None:
            route_steer, target_speed_throttle, target_speed_brake = (
                self.execute_route_and_target_speed(
                    open_loop_prediction.pred_route,
                    open_loop_prediction.pred_target_speed_scalar,
                    ego_speed,
                )
            )
        if open_loop_prediction.pred_future_waypoints is not None:
            waypoints_steer, waypoints_throttle, waypoints_brake = (
                self.execute_waypoints(
                    open_loop_prediction.pred_future_waypoints,
                    ego_speed,
                )
            )

        # Select which high-level modality we want to use for each control
        if self.config_closed_loop.steer_modality == "route":
            steer = route_steer
        elif self.config_closed_loop.steer_modality == "waypoint":
            steer = waypoints_steer
        else:
            raise ValueError(
                f"Invalid steer_modality: {self.config_closed_loop.steer_modality}",
            )

        if self.config_closed_loop.throttle_modality == "target_speed":
            throttle = target_speed_throttle
        elif self.config_closed_loop.throttle_modality == "waypoint":
            throttle = waypoints_throttle
        else:
            raise ValueError(
                f"Invalid throttle_modality: {self.config_closed_loop.throttle_modality}",
            )

        if self.config_closed_loop.brake_modality == "target_speed":
            brake = target_speed_brake
        elif self.config_closed_loop.brake_modality == "waypoint":
            brake = waypoints_brake
        else:
            raise ValueError(
                f"Invalid brake_modality: {self.config_closed_loop.brake_modality}",
            )

        # Turn off throttle if we brake
        if brake > 0.0:
            throttle = 0.0
            if (
                ego_speed < 0.01
            ):  # When we don't move we don't want the angle error to accumulate in the integral
                steer = 0.0

        # Return all predictions
        return ClosedLoopPrediction(
            **vars(open_loop_prediction),
            steer=steer,
            throttle=throttle,
            brake=brake,
            waypoints_steer=waypoints_steer,
            waypoints_throttle=waypoints_throttle,
            waypoints_brake=waypoints_brake,
            route_steer=route_steer,
            target_speed_throttle=target_speed_throttle,
            target_speed_brake=target_speed_brake,
            steering_alpha=steering_alpha,
        )
