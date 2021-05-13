"""
Script that prepares data for later usage

"""

from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
import bagpy

from scipy.interpolate.interpolate import interp1d
from pandas import DataFrame


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
    columns_except_time = data.columns.difference(["Time"])

    for col_name in columns_except_time:
        f = interp1d(shifted_timestamps.values, data[col_name].values)
        new_columns.append(pd.Series(f(timesteps), name=col_name))

    data_new = pd.concat(new_columns, axis=1)

    if plot_col:
        SAVEDIR = Path("plots/interpolation")
        sea.set_style("white")
        plt.figure(figsize=(5, 2.5))
        sea.lineplot(x=shifted_timestamps.values, y=data[plot_col], label="original")
        sea.lineplot(x="Time", y=plot_col, data=data_new, label="interpolated")
        plt.ylabel("Velocity")
        plt.savefig(SAVEDIR.joinpath("%s.eps" % plot_col))

    return data_new


def bag_to_dataframe(bagpath: Path, topics: List[str]) -> Dict[str, DataFrame]:
    bag = bagpy.bagreader(bagpath)
    dataframes = dict()

    for topic in topics:
        topic_msgs = bag.message_by_topic(topic)
        dataframes[topic] = pd.read_csv(topic_msgs)

    return dataframes


def sychronize_bag_measurements(bagpath: Path, topics: List[str]):
    pass


def plot_surge(bagpath: Path) -> None:
    TAU = "/thrust/tau_delivered"
    ODOM = "/odometry/filtered"
    topics = [TAU, ODOM]

    dataframes = bag_to_dataframe(bagpath, topics)

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

    dataframes = bag_to_dataframe(bagpath, topics)

    sea.set_style("white")
    fig, axes = plt.subplots(3, 1, sharex=True)
    sea.lineplot(ax=axes[0], x="Time", y="pose.pose.position.y", data=dataframes[ODOM])
    sea.lineplot(ax=axes[1], x="Time", y="twist.twist.linear.y", data=dataframes[ODOM])
    sea.lineplot(ax=axes[2], x="Time", y="force.y", data=dataframes[TAU])
    plt.savefig("results/sway-2021-04-27-04-51-27.png")


def save_bag_as_df_feather(bagpath: Path, savedir: Path, topics: List[str]) -> None:
    dataframes = bag_to_dataframe(str(bagpath), topics)
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


if __name__ == "__main__":
    main()
