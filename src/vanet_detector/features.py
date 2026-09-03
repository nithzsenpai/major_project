from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from .constants import BEHAVIOR_FEATURES, RELATION_FEATURES
from .io import flatten_message


def circular_difference_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def _add_temporal_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("send_time_ns", kind="stable").copy()
    time_seconds = group["send_time_ns"].to_numpy(np.float64) / 1e9
    x = group["sender_x"].to_numpy(np.float64)
    y = group["sender_y"].to_numpy(np.float64)
    speed = group["speed"].to_numpy(np.float64)
    acceleration = group["acceleration"].to_numpy(np.float64)
    heading = group["heading"].to_numpy(np.float64)

    delta_t = np.diff(time_seconds, prepend=np.nan)
    valid_dt = delta_t[np.isfinite(delta_t) & (delta_t > 0)]
    first_dt = float(np.median(valid_dt)) if valid_dt.size else 1.0
    delta_t[0] = first_dt
    delta_t = np.clip(np.nan_to_num(delta_t, nan=first_dt, posinf=10.0), 0.01, 10.0)

    delta_x = np.diff(x, prepend=x[0])
    delta_y = np.diff(y, prepend=y[0])
    displacement = np.hypot(delta_x, delta_y)
    implied_speed = displacement / delta_t
    previous_speed = np.r_[speed[0], speed[:-1]]
    previous_acceleration = np.r_[acceleration[0], acceleration[:-1]]
    average_reported_speed = 0.5 * (speed + previous_speed)
    position_speed_error = np.abs(implied_speed - average_reported_speed)

    delta_speed = speed - previous_speed
    expected_delta_speed = 0.5 * (acceleration + previous_acceleration) * delta_t
    speed_acceleration_error = np.abs(delta_speed - expected_delta_speed)
    delta_acceleration = acceleration - previous_acceleration
    jerk = delta_acceleration / delta_t

    movement_heading = np.degrees(np.arctan2(delta_y, delta_x)) % 360.0
    stationary = displacement < 0.25
    movement_heading[stationary] = heading[stationary]
    heading_motion_error = circular_difference_degrees(heading, movement_heading)
    heading_motion_error[stationary] = 0.0
    previous_heading = np.r_[heading[0], heading[:-1]]
    signed_heading_delta = (heading - previous_heading + 180.0) % 360.0 - 180.0
    yaw_rate = signed_heading_delta / delta_t

    rx = group["receiver_x"].to_numpy(np.float64)
    ry = group["receiver_y"].to_numpy(np.float64)
    relative_x = x - rx
    relative_y = y - ry
    receiver_distance = np.hypot(relative_x, relative_y)
    relative_bearing = np.arctan2(relative_y, relative_x)
    receive_latency = (
        group["receive_time_ns"].to_numpy(np.float64)
        - group["send_time_ns"].to_numpy(np.float64)
    ) / 1e9

    raw_rssi = group["rssi_dbm"].to_numpy(np.float64)
    rssi_available = np.isfinite(raw_rssi).astype(np.float64)
    rssi_series = pd.Series(raw_rssi)
    rssi_filled_for_rolling = rssi_series.ffill()
    rssi_delta = rssi_filled_for_rolling.diff().fillna(0.0).to_numpy(dtype=np.float64, copy=True)
    rssi_rolling_mean = (
        rssi_series.rolling(window=5, min_periods=1).mean().fillna(0.0).to_numpy(np.float64)
    )
    rssi_rolling_std = (
        rssi_series.rolling(window=5, min_periods=2).std().fillna(0.0).to_numpy(np.float64)
    )
    rssi_value = np.nan_to_num(raw_rssi, nan=0.0)
    rssi_delta[rssi_available == 0] = 0.0

    values = {
        "heading_sin": np.sin(np.deg2rad(heading)),
        "heading_cos": np.cos(np.deg2rad(heading)),
        "delta_t": delta_t,
        "receive_latency": receive_latency,
        "delta_x": delta_x,
        "delta_y": delta_y,
        "displacement": displacement,
        "implied_speed": implied_speed,
        "position_speed_error": position_speed_error,
        "delta_speed": delta_speed,
        "expected_delta_speed": expected_delta_speed,
        "speed_acceleration_error": speed_acceleration_error,
        "delta_acceleration": delta_acceleration,
        "jerk": jerk,
        "movement_heading_sin": np.sin(np.deg2rad(movement_heading)),
        "movement_heading_cos": np.cos(np.deg2rad(movement_heading)),
        "heading_motion_error": heading_motion_error,
        "yaw_rate": yaw_rate,
        "receiver_distance": receiver_distance,
        "relative_bearing_sin": np.sin(relative_bearing),
        "relative_bearing_cos": np.cos(relative_bearing),
        "message_rate": 1.0 / delta_t,
        "history_valid": np.ones(len(group), dtype=np.float64),
        "sender_observation_log": np.log1p(np.arange(1, len(group) + 1)),
        "relation_history_valid": np.ones(len(group), dtype=np.float64),
        "rssi_available": rssi_available,
        "rssi_dbm": rssi_value,
        "rssi_delta": rssi_delta,
        "rssi_rolling_mean": rssi_rolling_mean,
        "rssi_rolling_std": rssi_rolling_std,
    }
    for name, array in values.items():
        group[name] = array
    return group


def _add_relation_features(frame: pd.DataFrame, bucket_seconds: float) -> pd.DataFrame:
    frame = frame.copy()
    defaults = {
        "active_sender_count": 1.0,
        "near_ids_1m": 0.0,
        "near_ids_3m": 0.0,
        "near_ids_10m": 0.0,
        "min_neighbor_distance": 300.0,
        "max_position_similarity": 0.0,
        "max_kinematic_similarity": 0.0,
        "max_motion_similarity": 0.0,
        "same_motion_count": 0.0,
        "near_distinct_alias_count": 0.0,
        "max_nearby_rssi_similarity": 0.0,
    }
    for name, value in defaults.items():
        frame[name] = value

    bucket = np.rint((frame["send_time_ns"].to_numpy(np.float64) / 1e9) / bucket_seconds)
    frame["_time_bucket"] = bucket.astype(np.int64)

    for _, indices in frame.groupby("_time_bucket", sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=np.int64)
        if idx.size < 2:
            continue
        part = frame.loc[idx]
        x = part["sender_x"].to_numpy(np.float64)
        y = part["sender_y"].to_numpy(np.float64)
        speed = part["speed"].to_numpy(np.float64)
        acceleration = part["acceleration"].to_numpy(np.float64)
        heading = part["heading"].to_numpy(np.float64)
        dx = part["delta_x"].to_numpy(np.float64)
        dy = part["delta_y"].to_numpy(np.float64)
        sender = part["sender_id"].astype(str).to_numpy()
        alias = part["sender_alias"].astype(str).to_numpy()
        rssi = part["rssi_dbm"].to_numpy(np.float64)
        rssi_available = part["rssi_available"].to_numpy(np.float64) > 0.5

        distance = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
        speed_difference = np.abs(speed[:, None] - speed[None, :])
        acceleration_difference = np.abs(
            acceleration[:, None] - acceleration[None, :]
        )
        heading_difference = circular_difference_degrees(
            heading[:, None], heading[None, :]
        )
        motion_difference = np.hypot(
            dx[:, None] - dx[None, :], dy[:, None] - dy[None, :]
        )
        valid = sender[:, None] != sender[None, :]
        masked_distance = np.where(valid, distance, np.inf)

        position_similarity = np.where(valid, np.exp(-distance / 5.0), 0.0)
        kinematic_similarity = np.where(
            valid,
            np.exp(
                -speed_difference / 2.0
                - acceleration_difference / 2.0
                - heading_difference / 15.0
            ),
            0.0,
        )
        motion_similarity = np.where(valid, np.exp(-motion_difference / 3.0), 0.0)
        close_alias = valid & (distance < 3.0) & (alias[:, None] != alias[None, :])
        valid_rssi = (
            valid
            & rssi_available[:, None]
            & rssi_available[None, :]
            & np.isfinite(rssi[:, None])
            & np.isfinite(rssi[None, :])
        )
        rssi_similarity = np.where(
            valid_rssi,
            np.exp(-np.abs(rssi[:, None] - rssi[None, :]) / 3.0),
            0.0,
        )
        same_motion = (
            valid
            & (distance < 10.0)
            & (speed_difference < 1.0)
            & (heading_difference < 5.0)
            & (motion_difference < 3.0)
        )

        distinct_count = float(len(set(sender.tolist())))
        frame.loc[idx, "active_sender_count"] = distinct_count
        frame.loc[idx, "near_ids_1m"] = np.sum(valid & (distance < 1.0), axis=1)
        frame.loc[idx, "near_ids_3m"] = np.sum(valid & (distance < 3.0), axis=1)
        frame.loc[idx, "near_ids_10m"] = np.sum(valid & (distance < 10.0), axis=1)
        minimum = np.min(masked_distance, axis=1)
        frame.loc[idx, "min_neighbor_distance"] = np.where(
            np.isfinite(minimum), minimum, 300.0
        )
        frame.loc[idx, "max_position_similarity"] = np.max(position_similarity, axis=1)
        frame.loc[idx, "max_kinematic_similarity"] = np.max(
            kinematic_similarity, axis=1
        )
        frame.loc[idx, "max_motion_similarity"] = np.max(motion_similarity, axis=1)
        frame.loc[idx, "same_motion_count"] = np.sum(same_motion, axis=1)
        frame.loc[idx, "near_distinct_alias_count"] = np.sum(close_alias, axis=1)
        frame.loc[idx, "max_nearby_rssi_similarity"] = np.max(rssi_similarity, axis=1)

    return frame.drop(columns=["_time_bucket"])


def build_receiver_frame(
    messages: list[dict],
    receiver_id: str,
    scenario: str,
    source_class: str,
    split: str,
    relation_bucket_seconds: float = 1.0,
    rssi_lookup: dict[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    records = [
        flatten_message(message, receiver_id, scenario, source_class, split)
        for message in messages
    ]
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records)
    if rssi_lookup:
        external_rssi = np.asarray(
            [
                rssi_lookup.get((str(receiver), str(message)), np.nan)
                for receiver, message in zip(
                    frame["receiver_id"], frame["message_id"], strict=True
                )
            ],
            dtype=np.float64,
        )
        use_external = np.isfinite(external_rssi)
        frame.loc[use_external, "rssi_dbm"] = external_rssi[use_external]
    frame = frame.sort_values(["sender_id", "send_time_ns"], kind="stable")
    pieces = [_add_temporal_features(group) for _, group in frame.groupby("sender_id")]
    frame = pd.concat(pieces, ignore_index=True)
    frame = _add_relation_features(frame, relation_bucket_seconds)

    numeric_features = list(BEHAVIOR_FEATURES) + list(RELATION_FEATURES)
    frame[numeric_features] = frame[numeric_features].replace([np.inf, -np.inf], np.nan)
    frame[numeric_features] = frame[numeric_features].fillna(0.0)
    ranges = {
        "speed": (0.0, 100.0),
        "acceleration": (-30.0, 30.0),
        "delta_t": (0.01, 10.0),
        "receive_latency": (0.0, 10.0),
        "delta_x": (-500.0, 500.0),
        "delta_y": (-500.0, 500.0),
        "displacement": (0.0, 1000.0),
        "implied_speed": (0.0, 200.0),
        "position_speed_error": (0.0, 200.0),
        "delta_speed": (-100.0, 100.0),
        "expected_delta_speed": (-100.0, 100.0),
        "speed_acceleration_error": (0.0, 200.0),
        "delta_acceleration": (-60.0, 60.0),
        "jerk": (-100.0, 100.0),
        "heading_motion_error": (0.0, 180.0),
        "yaw_rate": (-360.0, 360.0),
        "receiver_distance": (0.0, 2000.0),
        "message_rate": (0.0, 100.0),
        "history_valid": (0.0, 1.0),
        "sender_observation_log": (0.0, 20.0),
        "relation_history_valid": (0.0, 1.0),
        "active_sender_count": (1.0, 500.0),
        "near_ids_1m": (0.0, 500.0),
        "near_ids_3m": (0.0, 500.0),
        "near_ids_10m": (0.0, 500.0),
        "min_neighbor_distance": (0.0, 1000.0),
        "same_motion_count": (0.0, 500.0),
        "near_distinct_alias_count": (0.0, 500.0),
        "rssi_available": (0.0, 1.0),
        "rssi_dbm": (-150.0, 0.0),
        "rssi_delta": (-50.0, 50.0),
        "rssi_rolling_mean": (-150.0, 0.0),
        "rssi_rolling_std": (0.0, 50.0),
        "max_nearby_rssi_similarity": (0.0, 1.0),
    }
    for name, (minimum, maximum) in ranges.items():
        frame[name] = frame[name].clip(minimum, maximum)
    return frame


def iter_sequences(
    frame: pd.DataFrame,
    seq_len: int,
    stride: int,
    max_gap_seconds: float,
    keep_attack_source_normals: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    if frame.empty:
        return
    behavior_names = list(BEHAVIOR_FEATURES)
    relation_names = list(RELATION_FEATURES)
    relation_history_index = relation_names.index("relation_history_valid")

    for _, sender_frame in frame.groupby("sender_id", sort=False):
        sender_frame = sender_frame.sort_values("send_time_ns", kind="stable")
        times = sender_frame["send_time_ns"].to_numpy(np.float64) / 1e9
        boundaries = np.flatnonzero(np.diff(times) > max_gap_seconds) + 1
        segments = np.split(np.arange(len(sender_frame)), boundaries)
        history_valid_index = behavior_names.index("history_valid")
        observation_index = behavior_names.index("sender_observation_log")
        for segment_indices in segments:
            if segment_indices.size == 0:
                continue
            segment = sender_frame.iloc[segment_indices]
            behavior = segment[behavior_names].to_numpy(np.float32)
            relation = segment[relation_names].to_numpy(np.float32)
            labels = segment["label"].to_numpy(np.int64)
            source_class = str(segment["source_class"].iloc[0])
            for end in range(0, len(segment), stride):
                label = int(labels[end])
                if source_class != "normal" and label == 0 and not keep_attack_source_normals:
                    continue
                start = max(0, end - seq_len + 1)
                behavior_window = behavior[start : end + 1].copy()
                relation_window = relation[start : end + 1].copy()
                missing = seq_len - len(behavior_window)
                if missing > 0:
                    behavior_padding = np.repeat(behavior_window[:1], missing, axis=0)
                    behavior_padding[:, history_valid_index] = 0.0
                    behavior_padding[:, observation_index] = 0.0
                    relation_padding = np.repeat(relation_window[:1], missing, axis=0)
                    relation_padding[:, relation_history_index] = 0.0
                    behavior_window = np.concatenate([behavior_padding, behavior_window], axis=0)
                    relation_window = np.concatenate([relation_padding, relation_window], axis=0)
                yield behavior_window, relation_window, label
