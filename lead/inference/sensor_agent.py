import dataclasses
import json
import logging
import os
import shutil
import typing
from collections import deque
from copy import deepcopy
from pathlib import Path

import carla
import cv2
import jaxtyping as jt
import matplotlib
import numpy as np
import numpy.typing as npt
import torch
from agents.navigation.local_planner import RoadOption
from beartype import beartype
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.common import common_utils
from lead.common.base_agent import BaseAgent
from lead.common.constants import TransfuserBoundingBoxClass
from lead.common.logging_config import setup_logging
from lead.common.route_planner import RoutePlanner
from lead.common.sensor_setup import av_sensor_setup
from lead.data_loader import carla_dataset_utils, training_cache
from lead.data_loader.carla_dataset_utils import rasterize_lidar
from lead.expert import expert_utils
from lead.inference.closed_loop_inference import (
    ClosedLoopInference,
    ClosedLoopPrediction,
)
from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.inference.infraction_recorder import InfractionRecorder
from lead.inference.video_recorder import VideoRecorder
from lead.training.config_training import TrainingConfig
from lead.visualization.visualizer import Visualizer

from activation_steering.feature_collectors.tfv6 import TFv6FeatureCollector
from depth_ttc_gate import DepthTTCClient
from vlm_gate import AsyncVLMGate
from activation_steering.policy import (
    depth_ttc_enabled_from_env,
    policy_from_env,
    DepthTTCPolicy,
    steering_override_enabled_from_env,
    classify_override_alpha,
    OverrideTrajectory,
)

matplotlib.use("Agg")  # non-GUI backend for headless servers

setup_logging()
LOG = logging.getLogger(__name__)

# Configure pytorch for maximum performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


def get_entry_point():  # dead: disable
    return "SensorAgent"


class SensorAgent(BaseAgent, autonomous_agent.AutonomousAgent):
    @beartype
    def setup(self, path_to_conf_file: str, route_index=None, __=None):  # pyright: ignore[reportIncompatibleMethodOverride]
        # Set up test time training default parameters
        self.config_closed_loop = ClosedLoopConfig()
        super().setup(sensor_agent=True)
        self.config_path = path_to_conf_file
        self.step = -1
        self.initialized = False
        self.device = torch.device("cuda:0")
        self.route_run_id = (
            str(route_index)
            if route_index is not None
            else os.environ.get("BENCHMARK_ROUTE_ID")
        )

        # Load the config saved during training
        if self.config_closed_loop.is_bench2drive:
            path_to_conf_file = path_to_conf_file.split("+")[0]
        with open(
            os.path.join(path_to_conf_file, "config.json"),
            encoding="utf-8",
        ) as f:
            json_config = f.read()
            json_config = json.loads(json_config)

        # Generate new config for the case that it has new variables.
        self.training_config = TrainingConfig(json_config)

        # Store training config in base class for Kalman filter decision
        # This is accessed by BaseAgent._use_kalman_filter()

        # Load model files
        self.closed_loop_inference = ClosedLoopInference(
            config_training=self.training_config,
            config_closed_loop=self.config_closed_loop,
            config_expert=self.config_expert,
            model_path=path_to_conf_file,
            device=self.device,
            prefix="model",
        )

        # Cast all ensemble sub-networks to match the training float type (e.g., bfloat16/float32)
        target_dtype = self.training_config.torch_float_type
        for net in self.closed_loop_inference.nets:
            net.to(dtype=target_dtype)

        # Feature Collector Setup
        self.save_features = self._env_flag("SAVE_FEATURES") or self._env_flag("SAVE_FUSED_FEATURES")
        features_root = os.environ.get("FEATURES_ROOT") or os.environ.get("FUSED_FEATURES_PATH")
        self.feature_collectors: list[TFv6FeatureCollector] = []

        print("[SensorAgent.setup] SAVE_FEATURES:", self.save_features)
        print("[SensorAgent.setup] FEATURES_ROOT:", features_root)

        if self.save_features:
            if features_root is not None:
                self.features_root = Path(features_root)
            elif self.config_closed_loop.save_path is not None:
                self.features_root = self.config_closed_loop.save_path / "fused_features"
            else:
                self.features_root = Path("fused_features")

            if self.route_run_id:
                self.features_root /= self.route_run_id

            for model_idx, net in enumerate(self.closed_loop_inference.nets):
                feature_collector = TFv6FeatureCollector(
                    output_root=self.features_root,
                    model_idx=model_idx,
                    save_fused=True,
                    decoder_layers=(4, 5, 6),
                )
                feature_collector.attach(net)
                self.feature_collectors.append(feature_collector)

                LOG.info("Saving features using %s to %s", self.feature_collectors[model_idx].__class__.__name__, self.features_root)

        self.activation_action_log = self._activation_action_log_path()
        print("[SensorAgent.setup] ACTIVATION_ACTION_LOG:", self.activation_action_log)
        if self.activation_action_log is not None:
            LOG.info("Saving activation actions to %s", self.activation_action_log)

        # Activation steering & Depth-TTC initialization
        self.depth_ttc_steering = depth_ttc_enabled_from_env()
        self.depth_ttc_client = None

        print("[SensorAgent.setup] DEPTH_TTC_STEERING:", self.depth_ttc_steering)
        if self.depth_ttc_steering:
            self.depth_ttc_client = DepthTTCClient.from_env()
            LOG.info("Depth TTC steering enabled: %s", self.depth_ttc_client.server_url)
            print(
                "[DepthTTC] inference interception enabled: "
                "run_step replaces the normal single forward pass with candidate "
                "forward passes, then uses the selected candidate for control",
                flush=True,
            )

        # STEERING_OVERRIDE (ablation): alpha is NOT injected into the network. It is turned into a
        # direct edit of the open-loop prediction (left/right: route+waypoint lateral, brake: target speed 0)
        # and, for brake, a hard brake=1 on the final control. Works together with Depth-TTC.
        self.steering_override = steering_override_enabled_from_env()
        self.override_traj = OverrideTrajectory.from_env() if self.steering_override else None
        # TFv6 route/waypoints are (x forward, y right) -> lateral axis 1. Set to 0 if the sign test shows
        # your checkpoint uses (x lateral, y forward).
        self.override_lateral_axis = int(os.environ.get("STEERING_OVERRIDE_TFV6_LATERAL_AXIS", 1))
        self._override_base = None  # (step, un-steered OpenLoopPrediction), shared by all candidates of a frame
        print("[SensorAgent.setup] STEERING_OVERRIDE:", self.steering_override)

        vlm_enabled = self._env_flag("VLM_STEERING") and not self.depth_ttc_steering
        self.activation_policy = policy_from_env(
            vlm_enabled=vlm_enabled,
            depth_ttc_client=self.depth_ttc_client,
            training_config=self.training_config,
        )
        LOG.info("Loaded Activation Policy: %s", self.activation_policy.__class__.__name__)

        # VLM gate: background thread that uploads front-camera frames (+ speed / command /
        # target point / ego history) to the Alpamayo server and exposes the latest decision.
        self.vlm_gate: AsyncVLMGate | None = None
        self.vlm_last_decision = None
        self.vlm_last_decision_key = None
        self.vlm_decision_start_frame = None
        self.vlm_decision_age_frames = None
        self.vlm_history_steps = int(os.environ.get("VLM_ALPAMAYO_HISTORY_STEPS", 16))
        self.vlm_history_stride = int(os.environ.get("VLM_ALPAMAYO_HISTORY_STRIDE_FRAMES", 2))
        # Only send requests to the VLM while ego speed (m/s) is above this value.
        # HOW TO DISABLE: set env VLM_MIN_SPEED_MPS=0 (or any value <= 0) -> request at every VLM_EVERY_N frames.
        # Side effect of the default (3.5): once the car is slower than 3.5 m/s (e.g. braked by the VLM itself)
        # no new decision is requested, and the last one expires after VLM_DECISION_TTL_FRAMES.
        self.vlm_min_speed = float(os.environ.get("VLM_MIN_SPEED_MPS", 3.5))
        # rgb_2 / original_rgb come straight from CARLA/cv2 in BGR order; AsyncVLMGate expects RGB.
        # Set VLM_INPUT_IS_BGR=0 if the front image is already RGB.
        self.vlm_input_is_bgr = os.environ.get("VLM_INPUT_IS_BGR", "1").lower() in {"1", "true", "t", "yes", "y"}
        self._vlm_history_warned = False
        print("[SensorAgent.setup] VLM_STEERING enabled:", vlm_enabled)
        print("[SensorAgent.setup] VLM_MIN_SPEED_MPS:", self.vlm_min_speed)
        if vlm_enabled:
            self.vlm_gate = AsyncVLMGate.from_env()
            self.vlm_gate.start()

        # Post-processing heuristics
        self.bb_buffer: deque = deque(maxlen=1)
        self.stop_sign_post_processor = StopSignPostProcessor(
            config=self.training_config,
            config_test_time=self.config_closed_loop,
            bb_buffer=self.bb_buffer,
        )
        self.force_move_post_processor = ForceMovePostProcessor(
            config=self.training_config,
            config_test_time=self.config_closed_loop,
            lidar_queue=self.lidar_pc_queue,
        )
        self.metric_info: dict = {}
        self.meters_travelled = 0.0

        # Infraction tracking
        self.infraction_recorder = InfractionRecorder(
            config_closed_loop=self.config_closed_loop,
            agent_name="SensorAgent",
        )
        # Keep a stable attribute for existing call sites.
        self.infractions_log = self.infraction_recorder.infractions_log

        self.track = autonomous_agent.Track.SENSORS

        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg is not installed or not found in PATH. Please install ffmpeg to use video compression.",
            )

        self.debug_image_saved_once = False

    @staticmethod
    def _env_flag(name: str) -> bool:
        return os.environ.get(name, "").lower() in {"1", "true", "t", "yes", "y"}

    @staticmethod
    def _depth_ttc_enabled_from_env() -> bool:
        policy_name = os.environ.get("ACTIVATION_POLICY", os.environ.get("STEERING_POLICY", "")).lower()
        return (
            os.environ.get("DEPTH_TTC_STEERING", "").lower() in {"1", "true", "t", "yes", "y"}
            or policy_name in ("depth_ttc", "depth-ttc", "depthttc")
        )

    def _depth_ttc_camera_image(self, input_data: dict) -> np.ndarray:
        """Return the front-camera image expected by the Depth-TTC policy."""
        # os.makedirs("debug_data", exist_ok=True)

        if not self.debug_image_saved_once and self.step > 20:
            self.debug_image_saved_once = True

            for i in range(1, 7):
                key = f"rgb_{i}"
                if key not in input_data:
                    continue
                image = input_data[key]

                # Adjust this depending on your actual sensor format.
                image = np.asarray(image)

                print(key, image.shape, image.dtype)
                np.save(f"debug_data/{key}.npy", image)

            for name in ["original_rgb", "rgb"]:
                if name in input_data:
                    image = np.asarray(input_data[name])
                    print(name, image.shape, image.dtype)
                    np.save(f"debug_data/{name}.npy", image)

            image = np.asarray(input_data["original_rgb"])
            front_width = self.training_config.camera_calibration[1]["width"]

            front_image = image[:, :front_width]
            print("front_image", front_image.shape, front_image.dtype)
            np.save("debug_data/front_image.npy", front_image)

        front_image = input_data.get("rgb_2")
        if front_image is not None:
            return np.asarray(front_image)

        image = np.asarray(input_data["original_rgb"])
        front_width = self.training_config.camera_calibration[1]["width"]
        # return image[:, :front_width]  # This is the left camera
        return image[:, front_width : 2 * front_width]  # This is the front camera

    def _vlm_front_image(self, input_data: dict) -> np.ndarray:
        """Front camera image for the VLM as an RGB uint8 HxWx3 array (no debug side effects)."""
        front_image = input_data.get("rgb_2")
        if front_image is None:
            image = np.asarray(input_data["original_rgb"])
            front_width = self.training_config.camera_calibration[1]["width"]
            front_image = image[:, front_width : 2 * front_width]
        front_image = np.asarray(front_image)
        if self.vlm_input_is_bgr:
            front_image = front_image[..., ::-1]
        return np.ascontiguousarray(front_image)

    @staticmethod
    def _vlm_command_id(input_data: dict) -> int:
        """RoadOption id (1..6) recovered from the one-hot command the model sees."""
        return int(np.argmax(np.asarray(input_data["command"]).reshape(-1))) + 1

    @staticmethod
    def _yaw_rotation_matrix(yaw: float) -> np.ndarray:
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        return np.asarray(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def _vlm_ego_history(self):
        """Ego history in the current ego frame for Alpamayo (native_traj / native_cot modes).

        Uses BaseAgent.ego_past_positions / ego_past_yaws (oldest -> current, already expressed in the
        current ego frame). Returns (xyz [1,1,T,3], rot [1,1,T,3,3]) as nested lists, or (None, None).
        Indexing matches the reference agent: T=vlm_history_steps samples, spaced vlm_history_stride frames
        apart, ending at the current frame; missing history is padded with the oldest available sample.
        """
        try:
            positions = self.ego_past_positions
            yaws = self.ego_past_yaws
        except Exception as exc:  # noqa: BLE001 - never let VLM history break driving
            if not self._vlm_history_warned:
                self._vlm_history_warned = True
                LOG.warning("VLM ego history unavailable (%s); sending None", exc)
            return None, None
        if len(positions) == 0 or len(yaws) == 0:
            return None, None

        steps, stride = self.vlm_history_steps, self.vlm_history_stride
        xyz = np.zeros((steps, 3), dtype=np.float32)
        rot = np.zeros((steps, 3, 3), dtype=np.float32)
        for k in range(steps):
            back = stride * (steps - 1 - k)
            pos_idx = max(0, len(positions) - 1 - back)
            yaw_idx = max(0, len(yaws) - 1 - back)
            xyz[k, :2] = positions[pos_idx]
            rot[k] = self._yaw_rotation_matrix(yaws[yaw_idx])
        return xyz.reshape(1, 1, steps, 3).tolist(), rot.reshape(1, 1, steps, 3, 3).tolist()

    @staticmethod
    def _alpha_active(activation_alpha) -> bool:
        values = SensorAgent._serialize_alpha(activation_alpha)
        if values is None:
            return False
        if isinstance(values, list):
            return any(abs(v) > 1e-6 for v in values)
        return abs(values) > 1e-6

    def _vlm_policy_alpha(self):
        """Fetch the newest VLM decision, compute its age in frames and map it to an alpha vector."""
        decision = self.vlm_gate.latest()
        age = None
        if decision is not None:
            decision_key = (decision.frame_id, decision.timestamp)
            if decision_key != self.vlm_last_decision_key:
                self.vlm_last_decision_key = decision_key
                self.vlm_decision_start_frame = self.step
            age = self.step - self.vlm_decision_start_frame
        self.vlm_last_decision = decision
        self.vlm_decision_age_frames = age

        alpha = self.activation_policy.alpha(self.step, decision, decision_age_frames=age)
        if self.vlm_gate.verbose:
            if decision is None:
                print(f"[VLMGate] use step={self.step} no completed decision yet", flush=True)
            elif self._alpha_active(alpha):
                print(
                    f"[VLMGate] use step={self.step} decision_frame={decision.frame_id} "
                    f"age={age} action={getattr(decision, 'action', 'none')} reason={decision.reason}",
                    flush=True,
                )
        return alpha

    def _vlm_submit_frame(self, input_data: dict, speed: float) -> None:
        """Feed the current frame to the VLM gate (submit if allowed, otherwise just buffer it)."""
        front_image = self._vlm_front_image(input_data)
        if self.vlm_min_speed <= 0.0 or speed > self.vlm_min_speed:
            history_xyz, history_rot = self._vlm_ego_history()
            self.vlm_gate.submit(
                frame_id=self.step,
                rgb_image=front_image,
                speed=speed,
                command=self._vlm_command_id(input_data),
                target_point=input_data["target_point"],
                ego_history_xyz=history_xyz,
                ego_history_rot=history_rot,
            )
        else:
            # Keep the frame buffer filled so later requests still get their past frames.
            self.vlm_gate.observe(self.step, front_image)

    def _vlm_log_fields(self) -> dict:
        if self.vlm_gate is None:
            return {}
        decision = self.vlm_last_decision
        return {
            "vlm_decision_frame": None if decision is None else int(decision.frame_id),
            "vlm_decision_age_frames": self.vlm_decision_age_frames,
            "vlm_action": None if decision is None else str(decision.action),
            "vlm_reason": None if decision is None else str(decision.reason)[:240],
            "vlm_error": None if decision is None else str(decision.error),
        }

    @staticmethod
    def _serialize_alpha(activation_alpha):
        if activation_alpha is None:
            return None
        if hasattr(activation_alpha, "detach"):
            return activation_alpha.detach().cpu().flatten().tolist()
        if isinstance(activation_alpha, np.ndarray):
            return activation_alpha.reshape(-1).tolist()
        if isinstance(activation_alpha, (list, tuple)):
            return [float(v) for v in activation_alpha]
        return float(activation_alpha)

    def _activation_action_log_path(self) -> Path | None:
        if self.config_closed_loop.save_path is None or not self.route_run_id:
            return None
        log_dir = self.config_closed_loop.save_path / self.route_run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "activation_actions.jsonl"

    def _get_feature_path(
        self,
        frame_idx: int,
        feature_name: str = "fused_features",
        model_idx: int | None = 0
    ) -> Path | None:
        if model_idx is None or len(self.feature_collectors) == 0:
            return None
        return self.feature_collectors[model_idx].resolve_feature_path(
            feature_dir=self.features_root,
            frame_idx=frame_idx,
            feature_name=feature_name,
            model_idx=model_idx,
        )

    def _set_planning_queries(self) -> None:
        """Save each ensemble member's post-transformer planner queries."""
        nets = self.closed_loop_inference.nets
        for model_idx, net in enumerate(nets):
            planning_decoder = getattr(net, "planning_decoder", None)
            queries = getattr(planning_decoder, "last_queries", None)
            if queries is None:
                continue
            self.feature_collectors[model_idx].set_fused_feature(queries)

    def _log_activation_action(
        self,
        frame_idx: int,
        model_idx: int | None,
        control: carla.VehicleControl,
        speed: float,
        pred_target_speed: float,
        stop_for_stop_sign: bool,
        activation_alpha=None,  # Added parameter
        extra: dict | None = None,
    ) -> None:
        if self.activation_action_log is None:
            return

        feature_path = self._get_feature_path(frame_idx, model_idx=model_idx)
        record = {
            "frame": int(frame_idx),
            "speed": float(speed),
            "pred_target_speed": float(pred_target_speed),
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
            "stop_for_stop_sign": bool(stop_for_stop_sign),
            "activation_alpha": self._serialize_alpha(activation_alpha),  # Modified from None
            "feature_path": str(feature_path) if feature_path is not None else None,
            "model_idx": model_idx,
        }
        if extra:
            record.update(extra)
        with self.activation_action_log.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------ STEERING_OVERRIDE
    def _apply_override_traj(self, prediction, side: str):
        """Keep the forward coordinate of route/waypoints, replace the lateral one by the template."""
        fwd = 1 - self.override_lateral_axis
        lat = self.override_lateral_axis

        def _swap(points):
            if points is None:
                return None
            p = points.clone()  # (1, N, 2)
            s = p[0, :, fwd].detach().float().cpu().numpy()
            offset = self.override_traj.lateral(s, side)
            p[0, :, lat] = torch.as_tensor(offset, device=p.device, dtype=p.dtype)
            return p

        return dataclasses.replace(
            prediction,
            pred_route=_swap(prediction.pred_route),
            pred_future_waypoints=_swap(prediction.pred_future_waypoints),
        )

    def _apply_override_brake(self, prediction):
        """Brake candidate: target speed -> 0, so the PID brakes and Depth-TTC scores a stopping candidate."""
        field_names = {f.name for f in dataclasses.fields(prediction)}
        changes = {}
        speed = getattr(prediction, "pred_target_speed_scalar", None)
        if speed is not None:
            changes["pred_target_speed_scalar"] = torch.zeros_like(speed)
        dist = getattr(prediction, "pred_target_speed_distribution", None)
        if dist is not None and "pred_target_speed_distribution" in field_names:
            one_hot = torch.zeros_like(dist)
            one_hot[..., 0] = 1.0  # ASSUMPTION: speed class 0 == 0 m/s
            changes["pred_target_speed_distribution"] = one_hot
        return dataclasses.replace(prediction, **changes)

    def _predict_for_alpha(self, data, alpha):
        """predict() for one alpha. In override mode alpha is not injected into the network;
        it becomes a direct edit of the (un-steered) open-loop prediction instead."""
        if not self.steering_override:
            return self.closed_loop_inference.predict(data, alpha)
        action = classify_override_alpha(alpha)
        # the un-steered forward is identical for every candidate of this frame -> run it once
        if self._override_base is None or self._override_base[0] != self.step:
            self._override_base = (self.step, self.closed_loop_inference.predict(data, None))
        prediction = self._override_base[1]
        if action in ("left", "right"):
            return self._apply_override_traj(prediction, action)
        if action == "brake":
            return self._apply_override_brake(prediction)
        return prediction

    @beartype
    def set_global_plan(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        global_plan_gps: typing.Any,
        privileged_org_dense_route_world_coord: list[
            tuple[carla.Transform, RoadOption]
        ],
    ):
        """Store the global plan for privileged logging .
        The dense world-coordinate route is stored as privileged information for
        offline logging and metric computation and must not be
        used by the driving policy. This is expected to be called by the
        leaderboard/runner before the scenario starts and before `_init`
        initialises components that consume the stored plan.

        Args:
            global_plan_gps: Global route waypoints in GPS space as provided by
                the leaderboard.
            privileged_org_dense_route_world_coord: Dense global route in world coordinates.
        """
        self.privileged_org_dense_route_world_coord = (
            privileged_org_dense_route_world_coord
        )
        LOG.info(
            "Set global plan with %d waypoints.",
            len(self.privileged_org_dense_route_world_coord),
        )
        super().set_global_plan(global_plan_gps, privileged_org_dense_route_world_coord)

    def set_scenario(self, scenario):
        """Set the scenario reference to track infractions.

        This should be called by the leaderboard after loading the scenario.
        """
        self.infraction_recorder.set_scenario(scenario)
        LOG.info("[SensorAgent] Scenario reference set for infraction tracking")

    def _init(self) -> None:
        # Get the hero vehicle and the CARLA world
        self._vehicle: carla.Actor = typing.cast(carla.Actor, CarlaDataProvider.get_hero_actor())
        self._world: carla.World = self._vehicle.get_world()

        # Set up video recorder
        self.video_recorder = VideoRecorder(
            config_closed_loop=self.config_closed_loop,
            vehicle=self._vehicle,
            world=self._world,
            step_counter=self.step,
            training_config=self.training_config,
        )

        self.set_weather()

        self.initialized = True

    def set_weather(self):
        weather_name = None

        if self.config_closed_loop.random_weather:
            weathers = self.config_expert.weather_settings.keys()
            weather_name = np.random.choice(list(weathers))

        if self.config_closed_loop.custom_weather is not None:
            weather_name = self.config_closed_loop.custom_weather

        if weather_name is not None:
            weather = carla.WeatherParameters(
                **self.config_expert.weather_settings[weather_name],
            )
            self._world.set_weather(weather)
            LOG.info(f"Set weather to: {weather_name}")
            # night mode
            vehicles: list[carla.Vehicle] = self._world.get_actors().filter("*vehicle*")
            if expert_utils.get_night_mode(weather):
                for vehicle in vehicles:
                    vehicle.set_light_state(
                        carla.VehicleLightState(
                            carla.VehicleLightState.Position
                            | carla.VehicleLightState.LowBeam,
                        ),
                    )
            else:
                for vehicle in vehicles:
                    # vehicle.set_light_state(carla.VehicleLightState.NONE)
                    vehicle.set_light_state(
                        carla.VehicleLightState(
                            carla.VehicleLightState.NONE
                        ))

    @beartype
    def sensors(self) -> list[dict]:
        return av_sensor_setup(
            config=self.training_config,
            lidar=True,
            radar=True,
            sensor_agent=True,
            perturbate=False,
            perturbation_rotation=0.0,
            perturbation_translation=0.0,
        )

    def check_infractions(self) -> None:
        """Check and record infractions for the current rollout step."""
        self.infraction_recorder.check_infractions(
            step=self.step,
            meters_travelled=self.meters_travelled,
        )

    @beartype
    def set_target_points(self, input_data: dict, pop_distance: float):
        """Defines local planning signals based on the input data.

        Args:
            input_data: The input data containing sensor information and state. Will be fed into model.
            pop_distance: Distance threshold to pop waypoints from the route planner.
        """
        planner: RoutePlanner = self.gps_waypoint_planners_dict[pop_distance]

        @beartype
        def transform(point: list[float]) -> jt.Float[npt.NDArray, " 2"]:
            # Use filtered or noisy position based on training config
            ego_position = (
                self.filtered_state[:2]
                if self.config_closed_loop.use_kalman_filter
                else input_data["noisy_state"][:2]
            )
            return common_utils.inverse_conversion_2d(
                np.array(point),
                np.array(ego_position),
                typing.cast(float, self.compass),
            )

        next_target_points = [tp[0].tolist() for tp in planner.route]
        next_commands = [int(planner.route[i][1]) for i in range(len(planner.route))]

        # Merge duplicate consecutive target points
        filtered_tp_list = []
        filtered_command_list = []
        for pt, cmd in zip(next_target_points, next_commands, strict=False):
            if (
                len(next_target_points) == 2
                or not filtered_tp_list
                or not np.allclose(pt[:2], filtered_tp_list[-1][:2])
            ):
                filtered_tp_list.append(pt)
                filtered_command_list.append(cmd)
        next_target_points = filtered_tp_list
        next_commands = filtered_command_list

        if len(next_target_points) > 2:
            input_data["target_point_next"] = transform(next_target_points[2][:2])
            input_data["target_point"] = transform(next_target_points[1][:2])
            input_data["target_point_previous"] = transform(next_target_points[0][:2])
        else:
            assert len(next_target_points) == 2
            input_data["target_point_next"] = transform(next_target_points[1][:2])
            input_data["target_point"] = transform(next_target_points[1][:2])
            input_data["target_point_previous"] = transform(next_target_points[0][:2])

        input_data["command"] = carla_dataset_utils.command_to_one_hot(next_commands[0])
        input_data["next_command"] = carla_dataset_utils.command_to_one_hot(
            next_commands[1],
        )

    @beartype
    @torch.inference_mode()
    def tick(self, input_data: dict, use_kalman_filter: bool = True) -> dict:
        """Pre-processes sensor data"""
        input_data = super().tick(
            input_data,
            use_kalman_filter=self.training_config.use_kalman_filter_for_gps,
        )

        # Simulate JPEG compression to avoid train-test mismatch
        rgb = input_data["rgb"]
        input_data["original_rgb"] = rgb.copy()
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        _, rgb = cv2.imencode(
            ".jpg",
            rgb,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.config_closed_loop.jpeg_quality],
        )
        rgb = cv2.imdecode(rgb, cv2.IMREAD_UNCHANGED)
        rgb = np.transpose(rgb, (2, 0, 1))
        input_data["rgb"] = rgb

        # Horizontal FOV reduction: crop left and right, then resize back
        if self.training_config.horizontal_fov_reduction > 0:
            if input_data["rgb"] is not None:  # (C, H, W)
                input_data["rgb"] = common_utils.fov_crop(
                    input_data["rgb"],
                    self.training_config.horizontal_fov_reduction,
                    chw=True,
                )
            if input_data["original_rgb"] is not None:  # (H, W, C)
                input_data["original_rgb"] = common_utils.fov_crop(
                    input_data["original_rgb"],
                    self.training_config.horizontal_fov_reduction,
                )

        # Cut cameras down to only used cameras
        if (
            self.training_config.num_used_cameras
            != self.training_config.num_available_cameras
        ):
            n = self.training_config.num_available_cameras
            for modality in ["rgb", "original_rgb"]:
                # rgb is (C, H, W), original_rgb is (H, W, C)
                width_axis = 1 if modality == "original_rgb" else 2
                w = input_data[modality].shape[width_axis] // n

                rgb_slices = []
                for i, use in enumerate(self.training_config.used_cameras):
                    if use:
                        s, e = i * w, (i + 1) * w
                        if modality == "original_rgb":
                            rgb_slices.append(input_data[modality][:, s:e])
                        else:
                            rgb_slices.append(input_data[modality][:, :, s:e])

                input_data[modality] = np.concatenate(rgb_slices, axis=width_axis)

        # Plan next target point and command.
        self.set_target_points(
            input_data,
            pop_distance=self.config_closed_loop.route_planner_min_distance,
        )
        if self.config_closed_loop.sensor_agent_pop_distance_adaptive:
            dense_points = (
                np.linalg.norm(
                    input_data["target_point"] - input_data["target_point_next"],
                )
                < 10.0
                and min(
                    np.linalg.norm(input_data["target_point_previous"]),
                    np.linalg.norm(input_data["target_point"]),
                )
                < 10.0
            )
            dense_points = dense_points or (
                np.linalg.norm(
                    input_data["target_point_previous"] - input_data["target_point"],
                )
                < 10.0
                and min(
                    np.linalg.norm(input_data["target_point_previous"]),
                    np.linalg.norm(input_data["target_point"]),
                )
                < 10.0
            )
            if dense_points:
                self.set_target_points(input_data, pop_distance=4.0)

        # Ignore the next target point if it's too far away
        if (
            self.config_closed_loop.sensor_agent_skip_distant_target_point
            and np.linalg.norm(input_data["target_point_next"])
            > self.config_closed_loop.sensor_agent_skip_distant_target_point_threshold
        ):
            # Skip the next target point if it's too far away
            input_data["target_point_next"] = input_data["target_point"]

        # Lidar input
        lidar = self.accumulate_lidar()
        # Use only part of the lidar history we trained on
        lidar = lidar[lidar[:, -1] < self.training_config.training_used_lidar_steps]

        # At inference time, simulate laspy quantization to avoid train-test mismatch
        lidar[:, 0] = (
            np.round(lidar[:, 0] / self.config_expert.point_precision_x)
            * self.config_expert.point_precision_x
        )
        lidar[:, 1] = (
            np.round(lidar[:, 1] / self.config_expert.point_precision_y)
            * self.config_expert.point_precision_y
        )
        lidar[:, 2] = (
            np.round(lidar[:, 2] / self.config_expert.point_precision_z)
            * self.config_expert.point_precision_z
        )

        # Convert to pseudo image
        input_data["rasterized_lidar"] = rasterize_lidar(
            config=self.training_config,
            lidar=lidar[:, :3],
        )[..., None]

        # Simulate training time compression to avoid train-test mismatch
        input_data["rasterized_lidar"] = training_cache.compress_float_image(
            input_data["rasterized_lidar"],
            self.training_config,
        )
        input_data["rasterized_lidar"] = training_cache.decompress_float_image(
            input_data["rasterized_lidar"],
        ).squeeze()[None, None]

        # Radar input preprocessing
        if self.training_config.use_radars:
            # Preprocess radar input using the same function as during training
            input_data["radar"] = np.concatenate(
                carla_dataset_utils.preprocess_radar_input(
                    self.training_config,
                    input_data,
                ),
                axis=0,
            )

        return input_data

    @beartype
    @torch.inference_mode()
    def run_step(self, input_data: dict, timestamp: float, __=None) -> carla.VehicleControl:
        self.step += 1

        if not self.initialized:
            self._init()
            self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            input_data = self.tick(input_data)
            return self.control

        # Update video recorder step and demo cameras
        if hasattr(self, "video_recorder"):
            self.video_recorder.update_step(self.step)
            self.video_recorder.move_demo_cameras_with_ego()

        # Need to run this every step for GPS filtering
        input_data = self.tick(input_data)

        # Transform the data into torch tensor comforting with data loader's format.
        input_data_tensors: dict[str, torch.Tensor] = {
            "rgb": torch.Tensor(input_data["rgb"]).to(self.device, dtype=torch.float32)[
                None
            ],
            "rasterized_lidar": torch.Tensor(input_data["rasterized_lidar"]).to(
                self.device,
                dtype=torch.float32,
            ),
            "target_point_previous": torch.Tensor(input_data["target_point_previous"])
            .to(self.device, dtype=torch.float32)
            .view(1, 2),
            "target_point": torch.Tensor(input_data["target_point"])
            .to(self.device, dtype=torch.float32)
            .view(1, 2),
            "target_point_next": (
                torch.Tensor(input_data["target_point_next"]).to(
                    self.device,
                    dtype=torch.float32,
                )
            ).view(1, 2),
            "speed": torch.Tensor([input_data["speed"]])
            .to(self.device, dtype=torch.float32)
            .view(1),
            "command": torch.Tensor(input_data["command"])
            .to(self.device, dtype=torch.float32)
            .view(1, 6),
            "next_command": torch.Tensor(input_data["next_command"])
            .to(self.device, dtype=torch.float32)
            .view(1, 6),
            "town": np.array([self._world.get_map().name.split("/")[-1]]),  # pyright: ignore[reportAssignmentType]
        }

        # Add radar data if available
        if self.training_config.use_radars and "radar" in input_data:
            input_data_tensors["radar"] = torch.Tensor(input_data["radar"]).to(
                self.device,
                dtype=torch.float32,
            )[None]

        # Save input log if need
        if (
            self.config_closed_loop.save_path is not None
            and self.config_closed_loop.produce_input_log
        ):
            torch.save(
                {
                    k: v.to(torch.device("cpu")) if isinstance(v, torch.Tensor) else v
                    for k, v in input_data_tensors.items()
                },
                os.path.join(
                    self.config_closed_loop.input_log_path,
                    str(self.step).zfill(5),
                )
                + ".pth",
            )

        self.closed_loop_inference.next_step()  # Increment the inference step counter

        # The policy owns everything alpha-related (single forward, or several candidates + selection).
        # predict() is side-effect free w.r.t. the PID controllers; control() runs once on the winner.
        if isinstance(self.activation_policy, DepthTTCPolicy):
            decision = self.activation_policy.decide(
                frame=self.step,
                speed=float(input_data["speed"]),
                predict=lambda alpha: self._predict_for_alpha(input_data_tensors, alpha),
                get_image=lambda: self._depth_ttc_camera_image(input_data),
                input_data=input_data,
            )
            closed_loop_prediction: ClosedLoopPrediction = self.closed_loop_inference.control(
                data=input_data_tensors,
                open_loop_prediction=decision.prediction,
                steering_alpha=decision.alpha,
            )
            steering_alpha = closed_loop_prediction.steering_alpha
        else:
            if self.vlm_gate is not None:
                # Latest VLM decision (+ its age) -> alpha, then hand the current frame to the VLM.
                steering_alpha = self._vlm_policy_alpha()
                self._vlm_submit_frame(input_data, speed=float(input_data["speed"]))
            else:
                steering_alpha = self.activation_policy.alpha(self.step)
            if self.steering_override:
                closed_loop_prediction = self.closed_loop_inference.control(
                    data=input_data_tensors,
                    open_loop_prediction=self._predict_for_alpha(input_data_tensors, steering_alpha),
                    steering_alpha=steering_alpha,  # only logged, never injected
                )
            else:
                closed_loop_prediction = self.closed_loop_inference.forward(
                    data=input_data_tensors,
                    steering_alpha=steering_alpha
                )

        override_action = classify_override_alpha(steering_alpha) if self.steering_override else None

        # Collect and save intermediate features via FeatureCollector
        if self.save_features and len(self.feature_collectors) > 0:
            self._set_planning_queries()
            for model_idx, net in enumerate(self.closed_loop_inference.nets):
                self.feature_collectors[model_idx].collect(
                    model=net,
                    frame_idx=self.step,
                    model_idx=model_idx,
                )

        # Update bounding boxes
        if (
            closed_loop_prediction.pred_bounding_box_vehicle_system is not None
            and len(closed_loop_prediction.pred_bounding_box_vehicle_system) > 0
        ):
            self.bb_buffer.append(
                closed_loop_prediction.pred_bounding_box_vehicle_system,
            )

        # Post-processing heuristic
        self.stop_sign_post_processor.update_stop_box(
            self.ego_past_positions[-2][0],
            self.ego_past_positions[-2][1],
            self.ego_past_yaws[-2],
            0.0,
            0.0,
            0.0,
        )
        closed_loop_prediction.throttle, closed_loop_prediction.brake = (
            self.force_move_post_processor.adjust(
                input_data["speed"].item(),
                closed_loop_prediction.throttle,
                closed_loop_prediction.brake,
            )
        )
        closed_loop_prediction.throttle, closed_loop_prediction.brake = (
            self.stop_sign_post_processor.adjust(
                input_data["speed"].item(),
                closed_loop_prediction.throttle,
                closed_loop_prediction.brake,
            )
        )
        stop_for_stop_sign = bool(
            len(self.stop_sign_post_processor.stop_sign_buffer) > 0
            and closed_loop_prediction.brake
        )
        self.meters_travelled += (
            input_data["speed"].item() * self.config_closed_loop.carla_frame_rate
        )
        input_data["meters_travelled"] = self.meters_travelled

        self.control = carla.VehicleControl(
            steer=float(closed_loop_prediction.steer),
            throttle=float(closed_loop_prediction.throttle),
            brake=float(closed_loop_prediction.brake),
        )

        # CARLA will not let the car drive in the initial frames. This help the filter not get confused.
        if self.step < self.training_config.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)

        # STEERING_OVERRIDE brake: applied last so ForceMove / stop-sign post-processors cannot undo it.
        if override_action == "brake":
            self.control = carla.VehicleControl(steer=self.control.steer, throttle=0.0, brake=1.0)

        pred_target_speed = closed_loop_prediction.pred_target_speed_scalar
        pred_target_speed_value = (
            float(pred_target_speed.reshape(-1)[0].item())
            if pred_target_speed is not None
            else 0.0
        )

        if os.environ.get("DEBUG_LOG_EGO_STAT", "").lower() in (
            "1",
            "true",
            "t",
            "yes",
            "y",
        ):
            print(
                "[EgoStat] "
                f"frame={self.step} "
                f"speed={float(input_data['speed'].item()):.4f} "
                f"brake={float(self.control.brake):.4f} "
                f"steer={float(self.control.steer):.4f} "
                f"throttle={float(self.control.throttle):.4f} "
                f"pred_target_speed={pred_target_speed_value:.4f}",
                flush=True,
            )

        # Always write activation_actions.jsonl (needs config_closed_loop.save_path + route id).
        # Without feature collectors there is a single record per step (model_idx=None, feature_path=None);
        # with feature collectors there is one record per ensemble member, as before.
        log_model_indices = list(range(len(self.feature_collectors))) or [None]
        for model_idx in log_model_indices:
            self._log_activation_action(
                frame_idx=self.step,
                model_idx=model_idx,
                control=self.control,
                speed=float(input_data["speed"].item()),
                pred_target_speed=pred_target_speed_value,
                stop_for_stop_sign=stop_for_stop_sign,
                activation_alpha=steering_alpha,
                extra={**self._vlm_log_fields(), "override_action": override_action},
            )

        # Check for infractions at this step
        self.check_infractions()

        # Visualization of prediction for debugging and video recording
        input_data_tensors.update(
            {
                "steer": torch.Tensor([self.control.steer]),
                "throttle": torch.Tensor([self.control.throttle]),
                "brake": torch.Tensor([self.control.brake]).bool(),
                "distance_to_stop_sign": torch.Tensor(
                    [
                        self.stop_sign_post_processor.stop_sign_buffer[0].norm
                        if len(self.stop_sign_post_processor.stop_sign_buffer) > 0
                        else np.inf,
                    ],
                ),
                "stuck_detector": torch.Tensor(
                    [int(self.force_move_post_processor.stuck_detector)],
                ).int(),
                "force_move": torch.Tensor(
                    [int(self.force_move_post_processor.force_move)],
                ).int(),
                "route_curvature": torch.Tensor(
                    [
                        common_utils.waypoints_curvature(
                            typing.cast(torch.Tensor, closed_loop_prediction.pred_route).squeeze(),
                        ),
                    ],
                ),
                "meters_travelled": torch.Tensor([self.meters_travelled]),
            },
        )

        # Save input images as PNG and video
        if (
            self.config_closed_loop.save_path is not None
            and self.step % self.config_closed_loop.produce_frame_frequency == 0
        ):
            # Get the RGB image for visualization (before JPEG compression)
            input_image = input_data["original_rgb"].copy()
            # Save input image and video using VideoRecorder
            if hasattr(self, "video_recorder"):
                self.video_recorder.save_input_image(input_image)
                self.video_recorder.save_input_video_frame(input_image)

        # Save demo images
        if (
            self.config_closed_loop.save_path is not None
            and self.step % self.config_closed_loop.produce_frame_frequency == 0
        ):
            # Get predicted route and waypoints (if available)
            pred_waypoints = (
                closed_loop_prediction.pred_future_waypoints[0]
                if closed_loop_prediction.pred_future_waypoints is not None
                else None
            )

            # Prepare target points dictionary for BEV visualization
            target_points = {
                "previous": input_data.get("target_point_previous"),
                "current": input_data.get("target_point"),
                "next": input_data.get("target_point_next"),
            }

            # Save demo cameras with visualization using VideoRecorder
            if hasattr(self, "video_recorder"):
                self.video_recorder.save_demo_cameras(pred_waypoints, target_points)
                # Save grid (demo + input stacked vertically) with planning visualization
                self.video_recorder.save_grid_image_and_video(
                    pred_waypoints=pred_waypoints,
                    target_points=target_points,
                )

        # Save abstract debug images
        if (
            self.config_closed_loop.save_path is not None
            and (
                self.config_closed_loop.produce_debug_video
                or self.config_closed_loop.produce_debug_image
            )
            and self.step % self.config_closed_loop.produce_frame_frequency == 0
        ):
            # Produce image
            image = Visualizer(
                config=self.training_config,
                data=input_data_tensors,
                prediction=closed_loop_prediction,
                config_test_time=self.config_closed_loop,
                test_time=True,
            ).visualize_inference_prediction()
            image = np.array(image).astype(np.uint8)

            # Add frame number to upper-right corner
            cv2.putText(
                image,
                f"Frame: {self.step}",
                (image.shape[1] - 240, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Save debug image and video using VideoRecorder
            if hasattr(self, "video_recorder"):
                self.video_recorder.save_debug_video_frame(image)
                self.video_recorder.save_debug_image(image)

        # Save metric info if in Bench2Drive mode
        if self.config_closed_loop.is_bench2drive and hasattr(self, "get_metric_info"):
            metric = self.get_metric_info()
            self.metric_info[self.step] = metric
            with open(
                f"{self.config_closed_loop.save_path}/metric_info.json",
                "w",
            ) as outfile:
                json.dump(self.metric_info, outfile, indent=4)
        return self.control

    def destroy(self, results=None):
        LOG.info(results)

        if getattr(self, "vlm_gate", None) is not None:
            self.vlm_gate.stop()

        # Clean up video recorder
        if hasattr(self, "video_recorder"):
            self.video_recorder.cleanup_and_compress()


class StopSignPostProcessor:
    """Heuristics to obey stop sign law."""

    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        config_test_time: ClosedLoopConfig,
        bb_buffer: deque,
    ):
        self.config = config
        self.config_test_time = config_test_time
        self.bb_buffer = bb_buffer
        self.stop_sign_buffer: deque = deque(maxlen=1)
        self.clear_stop_sign_cool_down = 0  # Counter if we recently cleared a stop sign
        self.slower_stop_sign_count = 0
        self.slower_for_stop_sign_cool_down = 0

    @beartype
    def adjust(self, ego_speed: float, current_throttle: float, current_brake: float):
        """Checks whether the car is intersecting with one of the detected stop signs"""
        if not self.config_test_time.slower_for_stop_sign or len(self.bb_buffer) == 0:
            # LOG.info("No bounding box")
            return current_throttle, current_brake

        if self.clear_stop_sign_cool_down > 0:
            self.clear_stop_sign_cool_down -= 1
        if self.slower_for_stop_sign_cool_down > 0:
            self.slower_for_stop_sign_cool_down -= 1
        stop_sign_stop_predicted = False

        for bb in self.bb_buffer[-1]:
            if bb.clazz == TransfuserBoundingBoxClass.STOP_SIGN:  # Stop sign detected
                # LOG.info("Stop sign detected.")
                self.stop_sign_buffer.append(bb)

        if len(self.stop_sign_buffer) > 0:
            # Check if we need to stop
            stop_box = self.stop_sign_buffer[0]
            stop_origin = carla.Location(x=stop_box.x, y=stop_box.y, z=0.0)
            stop_extent = carla.Vector3D(stop_box.w, stop_box.h, 1.0)
            stop_carla_box = carla.BoundingBox(stop_origin, stop_extent)
            stop_carla_box.rotation = carla.Rotation(0.0, np.rad2deg(stop_box.yaw), 0.0)

            stop_sign_distance = np.linalg.norm([stop_box.x, stop_box.y])
            boxes_intersect = (
                stop_sign_distance
                < self.config_test_time.slower_for_stop_sign_dist_threshold
            )
            if boxes_intersect and self.clear_stop_sign_cool_down <= 0:
                if ego_speed > 0.01:
                    # LOG.info("Stop sign intersection detected.")
                    stop_sign_stop_predicted = True
                else:
                    # LOG.info("Stop sign intersection detected but car is already stopped.")
                    # We have cleared the stop sign
                    stop_sign_stop_predicted = False
                    self.stop_sign_buffer.pop()
                    # Stop signs don't come in herds, so we know we don't need to clear one for a while.
                    self.clear_stop_sign_cool_down = (
                        self.config_test_time.slower_for_stop_sign_cool_down
                    )
                    self.slower_stop_sign_count = 0
            elif (
                self.slower_for_stop_sign_cool_down <= 0
                and stop_sign_distance
                < self.config_test_time.slower_for_stop_sign_dist_threshold
            ):
                # LOG.info("Stop sign in range for slower.")
                self.slower_stop_sign_count = (
                    self.config_test_time.slower_for_stop_sign_count
                )
                self.slower_for_stop_sign_cool_down = (
                    self.config_test_time.slower_for_stop_sign_cool_down
                )

        if len(self.stop_sign_buffer) > 0:
            # Remove boxes that are too far away
            if self.stop_sign_buffer[0].norm > abs(self.config.max_x_meter):
                # LOG.info("Stop sign removed")
                self.stop_sign_buffer.pop()

        if stop_sign_stop_predicted:
            # LOG.info("Stopping for stop sign.")
            current_throttle = 0.0
            current_brake = True

        if (
            self.config_test_time.slower_for_stop_sign
            and self.slower_stop_sign_count > 0
        ):
            # LOG.info("Slowing down for stop sign.")
            current_throttle = np.clip(
                current_throttle,
                0.0,
                self.config_test_time.slower_for_stop_sign_throttle_threshold,
            )
            self.slower_stop_sign_count -= 1

        return current_throttle, current_brake

    @beartype
    def update_stop_box(
        self,
        x: float,
        y: float,
        orientation: float,
        x_target: float,
        y_target: float,
        orientation_target: float,
    ):
        if not self.config_test_time.slower_for_stop_sign:
            return
        if len(self.stop_sign_buffer) != 0:
            self.stop_sign_buffer.append(
                self.stop_sign_buffer[0].update(
                    x,
                    y,
                    orientation,
                    x_target,
                    y_target,
                    orientation_target,
                ),
            )


class ForceMovePostProcessor:
    """Forces the agent to move after a certain time of being stuck."""

    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        config_test_time: ClosedLoopConfig,
        lidar_queue: deque,
    ):
        self.config = config
        self.config_test_time = config_test_time
        self.stuck_detector = 0
        self.force_move = 0
        self.lidar_buffer = lidar_queue

    @beartype
    def adjust(
        self,
        ego_speed: float,
        current_throttle: float,
        current_brake: float,
    ) -> tuple[float, float]:
        if not self.config_test_time.sensor_agent_creeping:
            return current_throttle, current_brake
        if (
            ego_speed < 0.1
        ):  # 0.1 is just an arbitrary low number to threshold when the car is stopped
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        # If last red light was encountered a long time ago, we can assume it was cleared
        stuck_threshold = self.config_test_time.sensor_agent_stuck_threshold

        if self.stuck_detector > stuck_threshold:
            self.force_move = self.config_test_time.sensor_agent_stuck_move_duration

        if self.force_move > 0:
            emergency_stop = False
            # safety check
            safety_box = deepcopy(self.lidar_buffer[-1])

            # z-axis
            safety_box = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
            safety_box = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]

            # y-axis
            safety_box = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
            safety_box = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]

            # x-axis
            safety_box = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
            safety_box = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]
            if len(safety_box) > 0:  # Checks if the List is empty
                emergency_stop = True
                LOG.info("Creeping overriden by safety box.")
            if not emergency_stop:
                LOG.info("Detected agent being stuck.")
                current_throttle = max(
                    self.config_test_time.sensor_agent_stuck_throttle,
                    current_throttle,
                )
                current_brake = 0.0
                self.force_move -= 1
            else:
                LOG.info("Forced moving stopped by safety box.")
                current_throttle = 0.0
                current_brake = 1.0
                self.force_move = self.config_test_time.sensor_agent_stuck_move_duration
        return current_throttle, current_brake


# if __name__ == "__main__":
#     sensor_agent = SensorAgent()
