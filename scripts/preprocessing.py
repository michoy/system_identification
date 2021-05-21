"""
Script that prepares data for later usage

"""


from functools import reduce
from operator import add
from pathlib import Path
from typing import Dict, List
from enum import Enum

import bagpy
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
import pandas as pd
import seaborn as sea
from pandas import DataFrame
from scipy.interpolate.interpolate import interp1d
from sklearn import preprocessing
from numba import jit, njit

from helper import (
    ETA_EULER_DOFS,
    Jq,
    get_eta,
    get_nu,
    get_tau,
    make_df,
    profile,
    DFKeys,
    ORIENTATIONS_EULER,
    ORIENTATIONS_QUAT,
    POSITIONS,
    LINEAR_VELOCITIES,
    ANGULAR_VELOCITIES,
    ETA_DOFS,
)


def recreate_sampling_times(
    data: DataFrame,
    step_length: float,
    start_time: float,
    end_time: float,
    plot_col=None,
) -> DataFrame:

    """
    Functions that transforms measurement data with samples taken it any (possibly irregular)
    sample rate and outputs the same measurements evenly spanced according to a given step length.

    data:           dataframe with numeric values that includes a 'Time' column
    step length:    desired time between each sample timestep
    duration:       amount of time covered by measurements in data
    plot_col:       name of column that should be plotted before and after (for vertification purposes)
    """

    first_time_in_df = data[DFKeys.TIME.value].iloc[0]
    if start_time < first_time_in_df:
        raise ValueError("start time cannot precede first time in df")

    get_shifted_time = lambda row: row[DFKeys.TIME.value] - start_time
    shifted_timestamps = data.apply(get_shifted_time, axis=1).rename(
        DFKeys.TIME.value, axis=1
    )

    duration = end_time - start_time
    timesteps = np.arange(0, duration, step_length)
    new_columns = [pd.Series(timesteps, name=DFKeys.TIME.value)]
    columns_except_time = data.columns.difference(
        [
            DFKeys.TIME.value,
            "child_frame_id",
            "header.frame_id",
            "header.seq",
            "header.stamp.nsecs",
            "header.stamp.secs",
            "pose.covariance",
            "twist.covariance",
            "pins_0",
            "pins_1",
            "pins_2",
            "pins_3",
            "pins_4",
            "pins_5",
            "pins_6",
            "pins_7",
        ]
    )

    for col_name in columns_except_time:
        f = interp1d(shifted_timestamps.values, data[col_name].values)
        new_columns.append(pd.Series(f(timesteps), name=col_name))

    data_new = pd.concat(new_columns, axis=1)

    if plot_col in data.columns:
        SAVEDIR = Path("results/interpolation")
        sea.set_style("white")
        # plt.figure(figsize=(5, 2.5))
        sea.lineplot(x=shifted_timestamps.values, y=data[plot_col], label="original")
        sea.lineplot(
            x=DFKeys.TIME.value, y=plot_col, data=data_new, label="interpolated"
        )
        # plt.ylabel("Velocity")
        # plt.savefig(SAVEDIR.joinpath("%s.eps" % plot_col))
        plt.show()

    return data_new


def bag_to_dataframes(bagpath: Path, topics: List[str]) -> Dict[str, DataFrame]:
    bag = bagpy.bagreader(str(bagpath))
    dataframes = dict()

    for topic in topics:
        topic_msgs: str = bag.message_by_topic(topic)
        dataframes[topic] = pd.read_csv(topic_msgs)
        columns_except_time = dataframes[topic].columns.difference([DFKeys.TIME.value])

    return dataframes


def bag_to_dataframe(
    bagpath: Path, topics: List[str], step_length: float, plot_col=None
) -> DataFrame:
    """Function for converting messages on topics in a bag into a single
    dataframe with equal timesteps. Uses 1d interpolation to synchronize
    topics.

    Args:
        bagpath (Path): path to bag file
        topics (List[str]): list of topics that should be converted
        step_length (float): length between timesteps in the new dataframe

    Returns:
        DataFrame: dataframe containing the desired topics
    """

    # convert bag to dataframes
    dataframes = bag_to_dataframes(bagpath, topics)

    # find global start and end times
    start_times = list()
    end_times = list()
    for topic in topics:
        df = dataframes[topic]
        start_times.append(df[DFKeys.TIME.value].iloc[0])
        end_times.append(df[DFKeys.TIME.value].iloc[-1])
    start_time = max(start_times)
    end_time = min(end_times)

    # give all dataframes equal timesteps
    synchronized_dataframes = []
    for topic in topics:
        df = recreate_sampling_times(
            dataframes[topic], step_length, start_time, end_time, plot_col=plot_col
        )
        synchronized_dataframes.append(df)

    # merge dataframes
    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on=[DFKeys.TIME.value], how="outer"),
        synchronized_dataframes,
    )

    return df_merged


def transform_to_NED(df: DataFrame) -> DataFrame:

    invert_columns = [
        DFKeys.FORCE_Y.value,
        DFKeys.FORCE_Z.value,
        DFKeys.TORQUE_Y.value,
        DFKeys.TORQUE_Z.value,
        DFKeys.POSITION_Y.value,
        DFKeys.POSITION_Z.value,
        DFKeys.SWAY.value,
        DFKeys.HEAVE.value,
    ]
    df[invert_columns] = df[invert_columns].apply(lambda x: x * (-1))

    def rotate_quat(row):
        w = row.orientation_w
        x = row.orientation_x
        y = row.orientation_y
        z = row.orientation_z

        q = Quaternion(w, x, y, z)
        q_unit = q.normalised
        q_rot = Quaternion(axis=[1.0, 0.0, 0.0], degrees=180)
        q_res: Quaternion = q_rot.rotate(q_unit)

        return pd.Series(
            {
                DFKeys.ORIENTATION_W.value: q_res.w,
                DFKeys.ORIENTATION_X.value: q_res.x,
                DFKeys.ORIENTATION_Y.value: q_res.y,
                DFKeys.ORIENTATION_Z.value: q_res.z,
            }
        )

    df[ORIENTATIONS_QUAT] = df[ORIENTATIONS_QUAT].apply(rotate_quat, axis=1)

    return df


def single_conversion(bag_path: Path, save_dir: Path, plot_col=None):
    TOPICS = [
        "/thrust/tau_delivered",
        "/odometry/filtered",
        "/auv/battery_level/system",
        "/pwm",
    ]
    STEP_LENGTH = 0.1

    df = bag_to_dataframe(bag_path, TOPICS, STEP_LENGTH, plot_col)

    df = df.rename(
        columns={
            "data": "voltage",
            "pose.pose.position.x": DFKeys.POSITION_X.value,
            "pose.pose.position.y": DFKeys.POSITION_Y.value,
            "pose.pose.position.z": DFKeys.POSITION_Z.value,
            "pose.pose.orientation.w": DFKeys.ORIENTATION_W.value,
            "pose.pose.orientation.x": DFKeys.ORIENTATION_X.value,
            "pose.pose.orientation.y": DFKeys.ORIENTATION_Y.value,
            "pose.pose.orientation.z": DFKeys.ORIENTATION_Z.value,
            "twist.twist.linear.x": DFKeys.SURGE.value,
            "twist.twist.linear.y": DFKeys.SWAY.value,
            "twist.twist.linear.z": DFKeys.HEAVE.value,
            "twist.twist.angular.x": DFKeys.ROLL_VEL.value,
            "twist.twist.angular.y": DFKeys.PITCH_VEL.value,
            "twist.twist.angular.z": DFKeys.YAW_VEL.value,
            "positive_width_us_0": "pwm_0",
            "positive_width_us_1": "pwm_1",
            "positive_width_us_2": "pwm_2",
            "positive_width_us_3": "pwm_3",
            "positive_width_us_4": "pwm_4",
            "positive_width_us_5": "pwm_5",
            "positive_width_us_6": "pwm_6",
            "positive_width_us_7": "pwm_7",
            "force.x": DFKeys.FORCE_X.value,
            "force.y": DFKeys.FORCE_Y.value,
            "force.z": DFKeys.FORCE_Z.value,
            "torque.x": DFKeys.TORQUE_X.value,
            "torque.y": DFKeys.TORQUE_Y.value,
            "torque.z": DFKeys.TORQUE_Z.value,
        }
    )

    df = transform_to_NED(df)

    save_path = (save_dir / bag_path.name).with_suffix(".csv")
    df.to_csv(save_path)


def add_euler_angles(df: pd.DataFrame, append_str="") -> pd.DataFrame:
    def map_to_euler(row):
        angles = Quaternion(
            w=row[DFKeys.ORIENTATION_W.value + append_str],
            x=row[DFKeys.ORIENTATION_X.value + append_str],
            y=row[DFKeys.ORIENTATION_Y.value + append_str],
            z=row[DFKeys.ORIENTATION_Z.value + append_str],
        ).yaw_pitch_roll
        return pd.Series(
            {
                DFKeys.YAW.value + append_str: angles[0],
                DFKeys.PITCH.value + append_str: angles[1],
                DFKeys.ROLL.value + append_str: angles[2],
            }
        )

    dofs = [dof + append_str for dof in ORIENTATIONS_EULER]
    df[dofs] = df.apply(
        map_to_euler,
        axis=1,
    )
    return df


def full_conversion(bag_dir, save_dir):
    for element in bag_dir.iterdir():
        if element.is_file() and element.suffix == ".bag":
            single_conversion(bag_path=element, save_dir=save_dir)


def single_integration_check(csv_path: Path, plot=False, save=False):
    df = pd.read_csv(csv_path)
    dt = df[DFKeys.TIME.value].iloc[1] - df[DFKeys.TIME.value].iloc[0]
    nu = get_nu(df)
    eta = get_eta(df)
    eta_integrated = np.zeros(eta.shape)

    # add integration
    eta_integrated[0] = eta[0]
    for i in range(len(nu) - 1):
        eta_integrated[i + 1] = eta_integrated[i] + Jq(eta[i]) @ nu[i] * dt
        # eta_integrated[i + 1][3:7] = preprocessing.normalize(
        #     [eta_integrated[i + 1][3:7]]
        # )  # fossen2021 alg 2.1
        q = Quaternion(eta_integrated[i + 1][3:7]).normalised
        eta_integrated[i + 1][3:7] = q.elements
    for dof, values in zip(ETA_DOFS, eta_integrated.T):
        df[dof + "_integrated"] = values

    # add derivation

    # add euler angles
    df = add_euler_angles(df)
    df = add_euler_angles(df, append_str="_integrated")

    # add difference
    for dof in ETA_DOFS:
        df[dof + "_difference_integrated"] = df.apply(
            lambda row: row[dof + "_integrated"] - row[dof], axis=1
        )

    save_dir = Path("results/integration_checks")

    if save:
        df.to_csv(save_dir / csv_path.name)

    if plot:

        for dof in ETA_DOFS:
            fig, axes = plt.subplots(2, 1, sharex=True)

            sea.lineplot(
                ax=axes[0], x=DFKeys.TIME.value, y=dof, data=df, label="Measured"
            )
            sea.lineplot(
                ax=axes[0],
                x=DFKeys.TIME.value,
                y=dof + "_integrated",
                data=df,
                label="Integrated",
            )
            sea.lineplot(
                ax=axes[1],
                x=DFKeys.TIME.value,
                y=dof + "_difference_integrated",
                data=df,
                label="Integrated - Measured",
            )

            axes[0].set_ylabel(dof)
            axes[1].set_ylabel("Difference")

            plt.savefig(save_dir.joinpath("%s-%s.eps" % (csv_path.stem, dof)))
            plt.close(fig)


def full_integration_check(df_dir: Path, plot=True):
    for element in df_dir.iterdir():
        if element.is_file() and element.suffix == ".csv":
            single_integration_check(csv_path=element, plot=plot)


if __name__ == "__main__":
    BAG_DIR = Path("data/raw")
    SAVE_DIR = Path("data/preprocessed")

    DF_DIR = Path("data/preprocessed")

    # full_integration_check(DF_DIR)
    # full_conversion(BAG_DIR, SAVE_DIR)
    single_integration_check(DF_DIR / "yaw-1.csv", plot=True)
