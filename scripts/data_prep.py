from pathlib import Path
from typing import List, Dict

import bagpy
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np


def bag_to_dataframe(bagpath: Path, topics: List[str]) -> Dict[str, DataFrame]:
    bag = bagpy.bagreader(bagpath)
    dataframes = dict()

    for topic in topics:
        topic_msgs = bag.message_by_topic(topic)
        dataframes[topic] = pd.read_csv(topic_msgs)

    return dataframes


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
    


if __name__ == "__main__":
    data_conversion("surge")
