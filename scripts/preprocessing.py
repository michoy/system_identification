"""
Script that prepares data for later usage

"""


from functools import reduce
from pathlib import Path
from typing import Dict, List

import bagpy
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
import pandas as pd
import seaborn as sea
from pandas import DataFrame
from scipy.interpolate.interpolate import interp1d
from numba import jit, njit

from helper import Jq, get_eta, get_nu, get_tau, make_df, profile


def double_line_plot(odometry: DataFrame, thrust: DataFrame, save_path: Path):
    sea.set()
    fig, axes = plt.subplots(2, 1, sharex=True)

    odometry = odometry.head(2000)
    thrust = thrust.head(2000)

    sea.lineplot(ax=axes[0], y=odometry["linear.x"], x=odometry["Time"])
    sea.lineplot(ax=axes[1], y=thrust["force.x"], x=thrust["Time"])

    plt.savefig(save_path)


def check_for_missing_values(data: DataFrame) -> None:
    if data.isnull().values.any():
        print("The dataset has %d missing values" % data.isnull().sum().sum())
        for col_name in data.columns:
            print(
                "Column %s has %d missing values"
                % (col_name, data[col_name].isnull().sum())
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

    first_time_in_df = data["Time"].iloc[0]
    if start_time < first_time_in_df:
        raise ValueError("start time cannot precede first time in df")

    get_shifted_time = lambda row: row["Time"] - start_time
    shifted_timestamps = data.apply(get_shifted_time, axis=1).rename("Time", axis=1)

    duration = end_time - start_time
    timesteps = np.arange(0, duration, step_length)
    new_columns = [pd.Series(timesteps, name="Time")]
    columns_except_time = data.columns.difference(
        [
            "Time",
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
        sea.lineplot(x="Time", y=plot_col, data=data_new, label="interpolated")
        # plt.ylabel("Velocity")
        # plt.savefig(SAVEDIR.joinpath("%s.eps" % plot_col))
        plt.show()

    return data_new


def bag_to_dataframes(bagpath: Path, topics: List[str]) -> Dict[str, DataFrame]:
    bag = bagpy.bagreader(str(bagpath))
    dataframes = dict()

    for topic in topics:
        topic_msgs = bag.message_by_topic(topic)
        dataframes[topic] = pd.read_csv(topic_msgs)
        columns_except_time = dataframes[topic].columns.difference(["Time"])

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
        start_times.append(df["Time"].iloc[0])
        end_times.append(df["Time"].iloc[-1])
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
        lambda left, right: pd.merge(left, right, on=["Time"], how="outer"),
        synchronized_dataframes,
    )

    return df_merged


def plot_surge(bagpath: Path) -> None:
    TAU = "/thrust/tau_delivered"
    ODOM = "/odometry/filtered"
    topics = [TAU, ODOM]

    dataframes = bag_to_dataframes(bagpath, topics)

    sea.set_style("white")
    fig, axes = plt.subplots(3, 1, sharex=True)
    sea.lineplot(ax=axes[0], x="Time", y="pose.pose.position.x", data=dataframes[ODOM])
    sea.lineplot(ax=axes[1], x="Time", y="twist.twist.linear.x", data=dataframes[ODOM])
    sea.lineplot(ax=axes[2], x="Time", y="force.x", data=dataframes[TAU])
    plt.savefig("results/surge-2021-04-27-04-08-38.png")


def plot_sway(bagpath: Path) -> None:
    TAU = "/thrust/tau_delivered"
    ODOM = "/odometry/filtered"
    topics = [TAU, ODOM]

    dataframes = bag_to_dataframes(bagpath, topics)

    sea.set_style("white")
    fig, axes = plt.subplots(3, 1, sharex=True)
    sea.lineplot(ax=axes[0], x="Time", y="pose.pose.position.y", data=dataframes[ODOM])
    sea.lineplot(ax=axes[1], x="Time", y="twist.twist.linear.y", data=dataframes[ODOM])
    sea.lineplot(ax=axes[2], x="Time", y="force.y", data=dataframes[TAU])
    plt.savefig("results/sway-2021-04-27-04-51-27.png")


def save_bag_as_df_feather(bagpath: Path, savedir: Path, topics: List[str]) -> None:
    dataframes = bag_to_dataframes(str(bagpath), topics)
    savedir = savedir / bagpath.stem
    savedir.mkdir(parents=True, exist_ok=True)
    for topic, dataframe in dataframes.items():
        topic_better_name = topic.replace("/", "-")[1:]
        savepath = savedir / topic_better_name
        dataframe.to_feather(savepath)


def data_conversion(test_name: str) -> None:
    bagdir = Path.home() / Path("data") / test_name
    savedir = Path.home() / Path("system_identification/data/raw") / test_name

    topics = [
        "/thrust/tau_delivered",
        "/odometry/filtered",
        "/thrust/desired_forces",
        "/auv/battery_level/system",
        "/imu/data_raw",
        "/dvl/dvl_msg",
        "/qualisys/Body_1/odom",
        "/pwm",
    ]

    for element in bagdir.iterdir():
        if element.is_file() and element.suffix == ".bag":
            save_bag_as_df_feather(bagpath=element, savedir=savedir, topics=topics)


def main():

    """Define paths for loading and saving dataframes"""

    RUN_NAME = "1D-run-1"
    ODOM_PATH = Path("data/%s/manta-pose_gt.csv" % RUN_NAME)
    THRUST_PATH = Path("data/%s/manta-thruster_manager-input.csv" % RUN_NAME)
    SAVEDIR = Path("data/processed")
    DEBUG_PLOT_DIR = Path("plots/debug")

    """ Load Dataframes """

    odometry = pd.read_csv(ODOM_PATH)
    thrust = pd.read_csv(THRUST_PATH)

    """ Check for missing values """

    check_for_missing_values(odometry)
    check_for_missing_values(thrust)

    """ Clean up dataframes """

    drop_columns = [
        "header.seq",
        "header.stamp.secs",
        "header.stamp.nsecs",
        "header.frame_id",
        "child_frame_id",
        "pose.covariance",
        "twist.covariance",
    ]
    odometry = odometry.drop(
        columns=drop_columns, axis="columns"
    ).rename(  # covariance is dropped because of interpolation difficulties
        columns={
            "pose.pose.position.x": "position.x",
            "pose.pose.position.y": "position.y",
            "pose.pose.position.z": "position.z",
            "pose.pose.orientation.x": "orientation.x",
            "pose.pose.orientation.y": "orientation.y",
            "pose.pose.orientation.z": "orientation.z",
            "pose.pose.orientation.w": "orientation.w",
            "twist.twist.linear.x": "linear.x",
            "twist.twist.linear.y": "linear.y",
            "twist.twist.linear.z": "linear.z",
            "twist.twist.angular.x": "angular.x",
            "twist.twist.angular.y": "angular.y",
            "twist.twist.angular.z": "angular.z",
        }
    )

    """ Synchronize sampling times """

    odom_init_time: float = odometry["Time"].iloc[0]
    odom_end_time: float = odometry["Time"].iloc[-1]
    thrust_init_time: float = thrust["Time"].iloc[0]
    thrust_end_time: float = thrust["Time"].iloc[-1]

    # define time boundry
    init_time = max(odom_init_time, thrust_init_time)
    end_time = min(odom_end_time, thrust_end_time)

    # only keep rows inside time boundry
    odometry = odometry[odometry["Time"] > init_time]
    odometry = odometry[odometry["Time"] < end_time]
    thrust = thrust[thrust["Time"] > init_time]
    thrust = thrust[thrust["Time"] < end_time]

    # create synchronized datasets through interpolation
    thrust = recreate_sampling_times(
        thrust, step_length=0.1, duration=end_time - init_time
    )
    odometry = recreate_sampling_times(
        odometry, step_length=0.1, duration=end_time - init_time, plot_col="linear.x"
    )

    """ Merge dataframes """

    data = odometry.merge(
        thrust, how="inner", left_on="Time", right_on="Time"
    ).set_index("Time")

    """ Save processed data """

    data.to_csv(SAVEDIR.joinpath("%s.csv" % RUN_NAME))
    data.to_pickle(SAVEDIR.joinpath("%s.pickle" % RUN_NAME))


def transform_to_NED(df: DataFrame) -> DataFrame:

    invert_columns = [
        "force.y",
        "force.z",
        "torque.y",
        "torque.z",
        "position_y",
        "position_z",
        "linear_y",
        "linear_z",
    ]
    df[invert_columns] = df[invert_columns].apply(lambda x: x * (-1))

    quat_cols = ["orientation_w", "orientation_x", "orientation_y", "orientation_z"]
    quats: np.ndarray = df[quat_cols].to_numpy()

    
    def rotate_quat(row):
        w = row.orientation_w
        x = row.orientation_x
        y = row.orientation_y
        z = row.orientation_z

        q = Quaternion(w, x, y, z)
        q_unit = q.normalised
        q_rot = Quaternion(axis=[1.0, 0.0, 0.0], degrees=180)
        q_res = q_rot.rotate(q_unit)

        return pd.Series(
            {
                "orientation_w": q_res.w,
                "orientation_x": q_res.x,
                "orientation_y": q_res.y,
                "orientation_z": q_res.z,
            }
        )

    df[quat_cols] = df[quat_cols].apply(rotate_quat, axis=1)

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
            "pose.pose.position.x": "position_x",
            "pose.pose.position.y": "position_y",
            "pose.pose.position.z": "position_z",
            "pose.pose.orientation.x": "orientation_x",
            "pose.pose.orientation.y": "orientation_y",
            "pose.pose.orientation.z": "orientation_z",
            "pose.pose.orientation.w": "orientation_w",
            "twist.twist.linear.x": "linear_x",
            "twist.twist.linear.y": "linear_y",
            "twist.twist.linear.z": "linear_z",
            "twist.twist.angular.x": "angular_x",
            "twist.twist.angular.y": "angular_y",
            "twist.twist.angular.z": "angular_z",
            "positive_width_us_0": "pwm_0",
            "positive_width_us_1": "pwm_1",
            "positive_width_us_2": "pwm_2",
            "positive_width_us_3": "pwm_3",
            "positive_width_us_4": "pwm_4",
            "positive_width_us_5": "pwm_5",
            "positive_width_us_6": "pwm_6",
            "positive_width_us_7": "pwm_7",
        }
    )

    df = transform_to_NED(df)

    save_path = (save_dir / bag_path.name).with_suffix(".csv")
    df.to_csv(save_path)


def full_conversion(bag_dir, save_dir):
    for element in bag_dir.iterdir():
        if element.is_file() and element.suffix == ".bag":
            single_conversion(bag_path=element, save_dir=save_dir)


def single_integration_check(csv_path: Path, plot=False, save=False):
    df = pd.read_csv(csv_path)
    dt = df["Time"].iloc[1] - df["Time"].iloc[0]
    nu = get_nu(df)
    eta = get_eta(df)
    eta_euler = np.zeros(eta.shape)

    eta_euler[0] = eta[0]
    for i in range(len(nu) - 1):
        eta_euler[i + 1] = eta_euler[i] + Jq(eta[i]) @ nu[i] * dt

    df_euler = make_df(df["Time"].to_numpy(), eta=eta_euler, nu=nu)
    
    if save:
        save_dir = Path("results/integration_checks")
        df_euler.to_csv(save_dir / csv_path.name)

    if plot:
        pos_dofs = ["position_x", "position_y", "position_z"]
        vel_dofs = ["linear_x", "linear_y", "linear_z"]
        for pos_key, vel_key in zip(pos_dofs, vel_dofs):
            fig, axes = plt.subplots(2, 1, sharex=True)

            sea.lineplot(ax=axes[0], x="Time", y=pos_key, data=df, label="Measured")
            sea.lineplot(ax=axes[0], x="Time", y=pos_key, data=df_euler, label="Euler")

            sea.lineplot(ax=axes[1], x="Time", y=vel_key, data=df, label="Measured")

            plt.savefig(save_dir.joinpath("%s-%s.eps" % (csv_path.stem, pos_key)))
            
def full_integration_check(df_dir: Path, plot=True):
    for element in df_dir.iterdir():
        if element.is_file() and element.suffix == ".csv":
            single_integration_check(csv_path=element, plot=plot)


if __name__ == "__main__":
    BAG_DIR = Path("/home/michaelhoyer/system_identification/data/raw")
    SAVE_DIR = Path("/home/michaelhoyer/system_identification/data/synchronized")

    DF_DIR = Path("data/synchronized")

    full_conversion(BAG_DIR, SAVE_DIR)
    full_integration_check(DF_DIR)
