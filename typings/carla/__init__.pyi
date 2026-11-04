"""
CARLA 0.9.15 Python API type stubs.

This is a hand-maintained typing surface for the commonly used CARLA API.
The actual implementations are provided by the compiled `carla.libcarla`
extension; this file exists only for Pyright/Pylance.

CARLA 0.9.15 / Python 3.10.
"""

from typing import Any, Callable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Math / geometry
# ---------------------------------------------------------------------------

class Vector2D:
    x: float
    y: float
    def __init__(self, x: float = ..., y: float = ...) -> None: ...
    def __add__(self, other: "Vector2D") -> "Vector2D": ...
    def __sub__(self, other: "Vector2D") -> "Vector2D": ...
    def __mul__(self, scalar: float) -> "Vector2D": ...
    def __truediv__(self, scalar: float) -> "Vector2D": ...
    def __eq__(self, other: object) -> bool: ...
    def length(self) -> float: ...
    def make_unit_vector(self) -> "Vector2D": ...
    def dot(self, other: "Vector2D") -> float: ...
    def __str__(self) -> str: ...


class Vector3D:
    x: float
    y: float
    z: float
    def __init__(self, x: float = ..., y: float = ..., z: float = ...) -> None: ...
    def __add__(self, other: "Vector3D") -> "Vector3D": ...
    def __sub__(self, other: "Vector3D") -> "Vector3D": ...
    def __mul__(self, scalar: float) -> "Vector3D": ...
    def __truediv__(self, scalar: float) -> "Vector3D": ...
    def __eq__(self, other: object) -> bool: ...
    def length(self) -> float: ...
    def make_unit_vector(self) -> "Vector3D": ...
    def dot(self, other: "Vector3D") -> float: ...
    def cross(self, other: "Vector3D") -> "Vector3D": ...
    def distance(self, other: "Vector3D") -> float: ...
    def __str__(self) -> str: ...


class Location(Vector3D):
    x: float
    y: float
    z: float
    def __init__(self, x: float = ..., y: float = ..., z: float = ...) -> None: ...
    def distance(self, other: Vector3D) -> float: ...
    def distance_2d(self, other: "Location") -> float: ...
    def __str__(self) -> str: ...


class Rotation:
    pitch: float
    yaw: float
    roll: float

    def __init__(
        self,
        pitch: float = ...,
        yaw: float = ...,
        roll: float = ...,
    ) -> None: ...

    def get_forward_vector(self) -> Vector3D: ...
    def get_right_vector(self) -> Vector3D: ...
    def get_up_vector(self) -> Vector3D: ...
    def __eq__(self, other: object) -> bool: ...
    def __str__(self) -> str: ...


class Transform:
    location: Location
    rotation: Rotation

    def __init__(
        self,
        location: Optional[Location] = ...,
        rotation: Optional[Rotation] = ...,
    ) -> None: ...

    def transform(self, in_point: Location) -> Location: ...
    def inverse_transform(self, in_point: Location) -> Location: ...
    def get_forward_vector(self) -> Vector3D: ...
    def get_right_vector(self) -> Vector3D: ...
    def get_up_vector(self) -> Vector3D: ...
    def __eq__(self, other: object) -> bool: ...
    def __str__(self) -> str: ...


class BoundingBox:
    location: Location
    rotation: Rotation
    extent: Vector3D

    def __init__(
        self,
        location: Location = ...,
        extent: Vector3D = ...,
        rotation: Optional[Rotation] = ...,
    ) -> None: ...

    def contains(self, world_point: Location) -> bool: ...
    def get_world_vertices(self, transform: Transform) -> List[Location]: ...
    def get_local_vertices(self) -> List[Location]: ...
    def __str__(self) -> str: ...


class Color:
    r: int
    g: int
    b: int
    a: int
    def __init__(self, r: int = ..., g: int = ..., b: int = ..., a: int = ...) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __str__(self) -> str: ...


# ---------------------------------------------------------------------------
# Actor / blueprint API
# ---------------------------------------------------------------------------

class ActorAttribute:
    id: str
    type: Any
    is_modifiable: bool
    recommended_values: List[str]
    def as_bool(self) -> bool: ...
    def as_int(self) -> int: ...
    def as_float(self) -> float: ...
    def as_str(self) -> str: ...
    def as_color(self) -> Color: ...
    def __str__(self) -> str: ...
    def __bool__(self) -> bool: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...


class ActorBlueprint:
    id: str
    tags: str
    def contains_tag(self, tag: str) -> bool: ...
    def match_tags(self, wildcard_pattern: str) -> bool: ...
    def has_attribute(self, id: str) -> bool: ...
    def has_tag(self, tag: str) -> bool: ...
    def get_attribute(self, id: str) -> ActorAttribute: ...
    def set_attribute(self, id: str, value: str) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[ActorAttribute]: ...
    def __getitem__(self, key: Union[int, str]) -> ActorAttribute: ...
    def __str__(self) -> str: ...


class BlueprintLibrary:
    def find(self, id: str) -> ActorBlueprint: ...
    def filter(self, wildcard_pattern: str) -> List[ActorBlueprint]: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[ActorBlueprint]: ...
    def __getitem__(self, pos: int) -> ActorBlueprint: ...
    def __str__(self) -> str: ...


class Actor:
    id: int
    type_id: str
    attributes: Mapping[str, ActorAttribute]
    bounding_box: BoundingBox
    semantic_tags: List[int]

    def get_world(self) -> "World": ...
    def get_location(self) -> Location: ...
    def get_transform(self) -> Transform: ...
    def get_velocity(self) -> Vector3D: ...
    def get_acceleration(self) -> Vector3D: ...
    def get_angular_velocity(self) -> Vector3D: ...
    def set_location(self, location: Location) -> None: ...
    def set_transform(self, transform: Transform) -> None: ...
    def set_target_velocity(self, velocity: Vector3D) -> None: ...
    def set_target_angular_velocity(self, angular_velocity: Vector3D) -> None: ...
    def set_simulate_physics(self, enabled: bool = ...) -> None: ...
    def set_enable_gravity(self, enabled: bool = ...) -> None: ...
    def destroy(self) -> bool: ...
    def is_alive(self) -> bool: ...
    def is_dormant(self) -> bool: ...
    def get_actor_state(self) -> Any: ...
    def listen(self, callback: Callable[[Any], None]) -> None: ...
    def stop(self) -> None: ...
    def __str__(self) -> str: ...


class ActorList:
    def __init__(self, actor_list):
        self.actor_list = actor_list

    def find(self, actor_id: int) -> Optional[Actor]: ...

    def filter(self, filterstring):
        if filterstring.startswith("vehicle"):
            return [
                actor
                for actor in self.actor_list
                if isinstance(actor, Vehicle)
            ]

        if filterstring.startswith("walker"):
            return [
                actor
                for actor in self.actor_list
                if isinstance(actor, Walker)
            ]

        return []

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Actor]: ...
    def __getitem__(self, pos: int) -> Actor: ...
    def __str__(self) -> str: ...


# ---------------------------------------------------------------------------
# Vehicle API
# ---------------------------------------------------------------------------

class VehicleControl:
    throttle: float
    steer: float
    brake: float
    hand_brake: bool
    reverse: bool
    manual_gear_shift: bool
    gear: int

    def __init__(
        self,
        throttle: float = ...,
        steer: float = ...,
        brake: float = ...,
        hand_brake: bool = ...,
        reverse: bool = ...,
        manual_gear_shift: bool = ...,
        gear: int = ...,
    ) -> None: ...

    def __eq__(self, other: object) -> bool: ...
    def __str__(self) -> str: ...


class VehicleLightState:
    NONE: int
    Position: int
    LowBeam: int
    HighBeam: int
    Brake: int
    RightBlinker: int
    LeftBlinker: int
    Reverse: int
    Fog: int
    Interior: int
    Special1: int
    Special2: int
    All: int
    def __init__(self, state: int = ...) -> None: ...
    def __or__(self, other: "VehicleLightState") -> "VehicleLightState": ...
    def __and__(self, other: "VehicleLightState") -> "VehicleLightState": ...


class VehicleWheel:
    position: Any
    tire_friction: float
    damping_rate: float
    max_steer_angle: float
    radius: float
    location: Location
    rotation: Rotation
    is_steerable: bool
    def __init__(self) -> None: ...


class VehicleWheelLocation:
    FL_Wheel: Any
    FR_Wheel: Any
    BL_Wheel: Any
    BR_Wheel: Any
    Front_Wheel: Any


class VehiclePhysicsControl:
    torque_curve: List[Tuple[float, float]]
    max_rpm: float
    moi: float
    damping_rate_full_throttle: float
    damping_rate_zero_throttle_clutch_engaged: float
    use_gear_autobox: bool
    gear_switch_time: float
    clutch_strength: float
    mass: float
    drag_coefficient: float
    center_of_mass: Vector3D
    steering_curve: List[Tuple[float, float]]
    wheels: List[VehicleWheel]
    def __init__(self, **kwargs: Any) -> None: ...
    def __str__(self) -> str: ...


class Vehicle(Actor):
    def apply_control(self, control: VehicleControl) -> None: ...
    def get_control(self) -> VehicleControl: ...
    def set_autopilot(self, enabled: bool = ..., tm_port: int = ...) -> None: ...
    def show_debug_telemetry(self, enabled: bool) -> None: ...
    def get_physics_control(self) -> VehiclePhysicsControl: ...
    def apply_physics_control(self, physics_control: VehiclePhysicsControl) -> None: ...
    def get_light_state(self) -> VehicleLightState: ...
    def set_light_state(self, light_state: VehicleLightState) -> None: ...
    def get_speed_limit(self) -> float: ...
    def open_door(self, door_idx: int) -> None: ...
    def close_door(self, door_idx: int) -> None: ...


# ---------------------------------------------------------------------------
# Walker API
# ---------------------------------------------------------------------------

class WalkerControl:
    direction: Vector3D
    speed: float
    jump: bool

    def __init__(
        self,
        direction: Optional[Vector3D] = ...,
        speed: float = ...,
        jump: bool = ...,
    ) -> None: ...


class Walker(Actor):
    def apply_control(self, control: WalkerControl) -> None: ...
    def get_control(self) -> WalkerControl: ...
    def get_bones_transform(self) -> Any: ...
    def set_bones(self, bones: Any) -> None: ...


# ---------------------------------------------------------------------------
# Sensor API / common sensor data
# ---------------------------------------------------------------------------

class Sensor(Actor):
    def listen(self, callback: Callable[[Any], None]) -> None: ...
    def is_listening(self) -> bool: ...
    def stop(self) -> None: ...


class CameraSensor(Sensor):
    pass


class Image:
    frame: int
    frame_number: int
    timestamp: float
    width: int
    height: int
    fov: float
    raw_data: Any
    def convert(self, color_converter: Any) -> None: ...
    def save_to_disk(self, path: str, color_converter: Any = ...) -> None: ...
    def __str__(self) -> str: ...


class ColorConverter:
    Raw: Any
    Depth: Any
    Depth_CityScapesPalette: Any
    CityScapesPalette: Any
    CityScapesPaletteLut: Any
    LogarithmicDepth: Any
    LogarithmicDepthNormalized: Any
    Segmentation: Any
    CityScapesPalette: Any
    def __init__(self, value: int = ...) -> None: ...


class LidarMeasurement:
    frame: int
    timestamp: float
    horizontal_angle: float
    channels: int
    raw_data: Any
    def save_to_disk(self, path: str) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Any]: ...


class LidarDetection:
    point: Location
    object_idx: int
    cos_inc_angle: float


class RadarDetection:
    altitude: float
    azimuth: float
    depth: float
    velocity: float
    def __init__(self) -> None: ...


class RadarMeasurement:
    frame: int
    timestamp: float
    horizontal_angle: float
    raw_data: Any
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[RadarDetection]: ...


class GnssMeasurement:
    frame: int
    timestamp: float
    latitude: float
    longitude: float
    altitude: float


class IMUMeasurement:
    frame: int
    timestamp: float
    accelerometer: Vector3D
    gyroscope: Vector3D
    compass: float


class CollisionEvent:
    frame: int
    timestamp: float
    other_actor: Actor
    normal_impulse: Vector3D


class LaneInvasionEvent:
    frame: int
    timestamp: float
    crossed_lane_markings: List[Any]


class ObstacleDetectionEvent:
    frame: int
    timestamp: float
    other_actor: Actor
    distance: float
    normal_impulse: Vector3D


# ---------------------------------------------------------------------------
# Map / road / waypoint API
# ---------------------------------------------------------------------------

class LaneType:
    None_: Any
    Driving: Any
    Stop: Any
    Shoulder: Any
    Biking: Any
    Sidewalk: Any
    Border: Any
    Restricted: Any
    Parking: Any
    Bidirectional: Any
    Median: Any
    Special1: Any
    Special2: Any
    Special3: Any
    Special4: Any
    Special5: Any
    Any: Any


class LaneChange:
    None_: Any
    Right: Any
    Left: Any
    Both: Any


class LaneMarkingType:
    None_: Any
    Other: Any
    Broken: Any
    Solid: Any
    SolidSolid: Any
    SolidBroken: Any
    BrokenSolid: Any
    BrokenBroken: Any
    BottsDots: Any


class LaneMarkingColor:
    Other: Any
    White: Any
    Yellow: Any
    Red: Any
    Blue: Any


class LaneMarking:
    type: Any
    color: Any
    width: float
    lane_change: Any


class Waypoint:
    id: int
    transform: Transform
    road_id: int
    section_id: int
    lane_id: int
    s: float
    lane_width: float
    lane_type: Any
    lane_change: Any
    left_lane_marking: LaneMarking
    right_lane_marking: LaneMarking

    def next(self, distance: float) -> List["Waypoint"]: ...
    def previous(self, distance: float) -> List["Waypoint"]: ...
    def next_until_lane_end(self, distance: float) -> List["Waypoint"]: ...
    def previous_until_lane_start(self, distance: float) -> List["Waypoint"]: ...
    def get_left_lane(self) -> Optional["Waypoint"]: ...
    def get_right_lane(self) -> Optional["Waypoint"]: ...
    def get_junction(self) -> "Junction": ...
    def is_junction(self) -> bool: ...

    def get_landmarks(
        self,
        distance: float,
        stop_at_junction: bool = ...,
    ) -> List["Landmark"]: ...

    def get_landmarks_of_type(
        self,
        distance: float,
        type: str,
        stop_at_junction: bool = ...,
    ) -> List["Landmark"]: ...

    def transform_to_traffic_light(self, traffic_light: "TrafficLight") -> Transform: ...
    def __eq__(self, other: object) -> bool: ...


class Junction:
    id: int
    bounding_box: BoundingBox
    def get_waypoints(self) -> List[Tuple[Waypoint, Waypoint]]: ...


class Map:
    name: str
    opendrive: str
    def get_spawn_points(self) -> List[Transform]: ...

    def get_waypoint(
        self,
        location: Location,
        project_to_road: bool = ...,
        lane_type: Any = ...,
    ) -> Optional[Waypoint]: ...

    def get_waypoint_xodr(
        self,
        road_id: int,
        lane_id: int,
        s: float,
    ) -> Optional[Waypoint]: ...

    def generate_waypoints(self, distance: float) -> List[Waypoint]: ...
    def get_topology(self) -> List[Tuple[Waypoint, Waypoint]]: ...
    def to_opendrive(self) -> str: ...
    def get_crosswalks(self) -> List[Location]: ...
    def get_all_landmarks_from_id(self, landmark_id: str) -> List["Landmark"]: ...


class Landmark:
    id: str
    name: str
    distance: float
    s: float
    type: str
    value: float
    orientation: str
    height: float
    width: float
    transform: Transform
    waypoint: Waypoint
    def get_waypoint(self) -> Waypoint: ...
    def get_transform(self) -> Transform: ...
    def set_traffic_light_state(self, state: Any) -> None: ...
    def set_distance(self, distance: float) -> None: ...


# ---------------------------------------------------------------------------
# World / simulation API
# ---------------------------------------------------------------------------

class Timestamp:
    frame: int
    elapsed_seconds: float
    delta_seconds: float
    platform_timestamp: float


class WorldSnapshot:
    frame: int
    timestamp: Timestamp
    def find(self, actor_id: int) -> Optional["ActorSnapshot"]: ...
    def __iter__(self) -> Iterator["ActorSnapshot"]: ...


class ActorSnapshot:
    id: int
    timestamp: Timestamp
    def get_transform(self) -> Transform: ...
    def get_velocity(self) -> Vector3D: ...
    def get_acceleration(self) -> Vector3D: ...
    def get_angular_velocity(self) -> Vector3D: ...
    def get_location(self) -> Location: ...


class Spectator(Actor):
    pass


class WorldSettings:
    synchronous_mode: bool
    no_rendering_mode: bool
    fixed_delta_seconds: Optional[float]
    substepping: bool
    max_substep_delta_time: float
    max_substeps: int
    max_culling_distance: float
    deterministic_ragdolls: bool
    tile_stream_distance: float
    actor_active_distance: float
    def __init__(self, **kwargs: Any) -> None: ...


class World:
    def get_map(self) -> Map: ...
    def get_blueprint_library(self) -> BlueprintLibrary: ...
    def get_actors(self, id: Optional[int] = ...) -> ActorList: ...
    def get_spectator(self) -> Spectator: ...

    def spawn_actor(
        self,
        blueprint: ActorBlueprint,
        transform: Transform,
        attach_to: Optional[Actor] = ...,
        attachment: Any = ...,
    ) -> Actor: ...

    def try_spawn_actor(
        self,
        blueprint: ActorBlueprint,
        transform: Transform,
        attach_to: Optional[Actor] = ...,
        attachment: Any = ...,
    ) -> Optional[Actor]: ...

    def get_settings(self) -> WorldSettings: ...
    def apply_settings(self, settings: WorldSettings, seconds: float = ...) -> int: ...
    def wait_for_tick(self, seconds: float = ...) -> WorldSnapshot: ...
    def on_tick(self, callback: Callable[[WorldSnapshot], None]) -> int: ...
    def remove_on_tick(self, callback_id: int) -> None: ...
    def tick(self, seconds: float = ...) -> int: ...
    def get_snapshot(self) -> WorldSnapshot: ...
    def get_weather(self) -> "WeatherParameters": ...
    def set_weather(self, weather: "WeatherParameters") -> None: ...
    def get_lightmanager(self) -> Any: ...
    def get_traffic_lights(self, location: Optional[Location] = ...) -> List["TrafficLight"]: ...

    def get_traffic_signs_in_distance(
        self,
        location: Location,
        distance: float,
    ) -> List["TrafficSign"]: ...

    def debug(self) -> Any: ...
    def get_random_location_from_navigation(self) -> Optional[Location]: ...
    def set_pedestrians_seed(self, seed: int) -> None: ...
    def set_pedestrians_cross_factor(self, percentage: float) -> None: ...


class Client:
    def __init__(
        self,
        host: str,
        port: int,
        worker_threads: int = ...,
    ) -> None: ...
    def set_timeout(self, seconds: float) -> None: ...
    def get_client_version(self) -> str: ...
    def get_server_version(self) -> str: ...
    def get_server_build_version(self) -> str: ...
    def get_world(self) -> World: ...

    def load_world(
        self,
        map_name: str,
        reset_settings: bool = ...,
        map_layers: Any = ...,
    ) -> World: ...

    def reload_world(self, reset_settings: bool = ...) -> World: ...
    def get_available_maps(self) -> List[str]: ...
    def apply_batch(self, commands: Sequence[Any], do_tick: bool = ...) -> List[Any]: ...

    def apply_batch_sync(
        self,
        commands: Sequence[Any],
        do_tick: bool = ...,
    ) -> List[Any]: ...

    def get_trafficmanager(self, port: int = ...) -> Any: ...
    def start_recorder(self, name: str, additional_data: bool = ...) -> None: ...
    def stop_recorder(self) -> None: ...
    def show_recorder_file_info(self, name: str, show_all: bool = ...) -> str: ...
    def show_recorder_collisions(self, name: str, category1: str, category2: str) -> str: ...
    def show_recorder_actors_blocked(self, name: str, min_time: float, min_distance: float) -> str: ...
    def set_replayer_time_factor(self, time_factor: float) -> None: ...

    def replay_file(
        self,
        name: str,
        start: float,
        duration: float,
        follow_id: int = ...,
        replay_sensors: bool = ...,
    ) -> None: ...

    def set_replayer_ignore_hero(self, ignore: bool) -> None: ...
    def set_replayer_ignore_spectator(self, ignore: bool) -> None: ...
    def set_replayer_ignore_vehicles(self, ignore: bool) -> None: ...
    def set_replayer_ignore_walkers(self, ignore: bool) -> None: ...
    def set_replayer_ignore_sensors(self, ignore: bool) -> None: ...
    def get_required_files(self) -> List[str]: ...


# ---------------------------------------------------------------------------
# Traffic lights / signs
# ---------------------------------------------------------------------------

class TrafficLightState:
    Red: Any
    Yellow: Any
    Green: Any
    Off: Any
    Unknown: Any


class TrafficLight(Actor):
    trigger_volume: BoundingBox
    def get_state(self) -> Any: ...
    def set_state(self, state: Any) -> None: ...
    def get_green_time(self) -> float: ...
    def set_green_time(self, green_time: float) -> None: ...
    def get_yellow_time(self) -> float: ...
    def set_yellow_time(self, yellow_time: float) -> None: ...
    def get_red_time(self) -> float: ...
    def set_red_time(self, red_time: float) -> None: ...
    def get_elapsed_time(self) -> float: ...
    def freeze(self, freeze: bool) -> None: ...
    def get_pole_index(self) -> int: ...
    def get_group_traffic_lights(self) -> List["TrafficLight"]: ...
    def get_stop_waypoints(self) -> List[Waypoint]: ...


class TrafficSign(Actor):
    trigger_volume: BoundingBox


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

class WeatherParameters:
    cloudiness: float
    precipitation: float
    precipitation_deposits: float
    wind_intensity: float
    sun_azimuth_angle: float
    sun_altitude_angle: float
    fog_density: float
    fog_distance: float
    wetness: float
    fog_falloff: float
    scattering_intensity: float
    mie_scattering_scale: float
    rayleigh_scattering_scale: float
    dust_storm: float

    ClearNoon: "WeatherParameters"
    ClearSunset: "WeatherParameters"
    CloudyNoon: "WeatherParameters"
    CloudySunset: "WeatherParameters"
    WetNoon: "WeatherParameters"
    WetSunset: "WeatherParameters"
    MidRainyNoon: "WeatherParameters"
    MidRainSunset: "WeatherParameters"
    WetCloudyNoon: "WeatherParameters"
    WetCloudySunset: "WeatherParameters"
    HardRainNoon: "WeatherParameters"
    HardRainSunset: "WeatherParameters"
    SoftRainNoon: "WeatherParameters"
    SoftRainSunset: "WeatherParameters"

    def __init__(self, **kwargs: Any) -> None: ...


# ---------------------------------------------------------------------------
# Attachments / enums commonly referenced by CARLA scripts
# ---------------------------------------------------------------------------

class AttachmentType:
    Rigid: Any
    SpringArm: Any
    SpringArmGhost: Any


class ActorAttributeType:
    Bool: Any
    Int: Any
    Float: Any
    String: Any


class TrafficLightGroup:
    pass


# CARLA exposes these names through the extension. They are intentionally
# permissive here because their exact Boost.Python enum/value representation
# varies and is not useful for ordinary static checking.
AttachmentType = AttachmentType
ActorAttributeType = ActorAttributeType
