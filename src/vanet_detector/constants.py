from __future__ import annotations

CLASS_NAMES = ("normal", "sybil", "illusion")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

ATTACK_TO_CLASS = {
    "trafficCongestionSybil": "sybil",
    "constantPositionOffset": "illusion",
    "randomPositionOffset": "illusion",
    "positionMirroring": "illusion",
}

SCENARIOS = ("highway_2", "urban_2", "highway_7", "urban_7")

BEHAVIOR_FEATURES = (
    "speed",
    "acceleration",
    "heading_sin",
    "heading_cos",
    "delta_t",
    "receive_latency",
    "delta_x",
    "delta_y",
    "displacement",
    "implied_speed",
    "position_speed_error",
    "delta_speed",
    "expected_delta_speed",
    "speed_acceleration_error",
    "delta_acceleration",
    "jerk",
    "movement_heading_sin",
    "movement_heading_cos",
    "heading_motion_error",
    "yaw_rate",
    "receiver_distance",
    "relative_bearing_sin",
    "relative_bearing_cos",
    "message_rate",
    "history_valid",
    "sender_observation_log",
)

RELATION_FEATURES = (
    "relation_history_valid",
    "active_sender_count",
    "near_ids_1m",
    "near_ids_3m",
    "near_ids_10m",
    "min_neighbor_distance",
    "max_position_similarity",
    "max_kinematic_similarity",
    "max_motion_similarity",
    "same_motion_count",
    "near_distinct_alias_count",
    "rssi_available",
    "rssi_dbm",
    "rssi_delta",
    "rssi_rolling_mean",
    "rssi_rolling_std",
    "max_nearby_rssi_similarity",
)

ZENODO_RECORD_ID = "19665762"
