'''
Script for exploring data from csv files.

The individual steps taken in this script have to be adapted 
to the data at hand, but the overall structure will stay the same. 

'''

from os import rename
from pathlib import Path
from matplotlib.pyplot import axis, ylabel

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def main():

    RUN_NAME = '1D-run-1'
    data_dir = Path('data').joinpath(RUN_NAME)

    print_meta = False
    plot = True
    print_tables = False

    ''' Load dataframes '''

    odom_path = data_dir.joinpath('manta-pose_gt.csv')
    thrust_path = data_dir.joinpath('manta-thruster_manager-input.csv')

    odometry = pd.read_csv(odom_path)
    thrust = pd.read_csv(thrust_path)


    ''' Look at meta-data '''

    if print_meta:
        print('Odometry:')
        print(odometry.columns)
        print(odometry.head())
        print(odometry.shape)

        print('Desired thrust:')
        print(thrust.columns)
        print(thrust.head())
        print(thrust.shape)


    ''' Create plots '''

    if plot:

        plot_dir = Path('plots').joinpath(RUN_NAME)
        sns.set()

        # trajectory plot
        sns.lineplot(x='pose.pose.position.y', y='pose.pose.position.x', sort=False, lw=1, data=odometry)  
        plt.axis('equal')   
        plt.savefig(plot_dir.joinpath('trajectory-plot.eps'))

        # linear velocity plot
        sns.lineplot(data=odometry[['twist.twist.linear.x', 'twist.twist.linear.y', 'twist.twist.angular.z']], dashes=False, palette='colorblind')
        plt.savefig(plot_dir.joinpath('linear-velocity-plot.eps'))

        # desired thrust plot
        sns.lineplot(data=thrust[['force.x', 'force.y', 'torque.z']], dashes=False, palette='colorblind')
        plt.savefig(plot_dir.joinpath('thrust-plot.eps'))


        # renaming for nicer plots
        odometry = odometry.rename(columns={
            'pose.pose.position.x': 'Position', 
            'twist.twist.linear.x': 'Velocity', 
        })
        thrust = thrust.rename(columns={
            'force.x': 'Thrust'
        })

        sns.set_style('white')
        fig, axes = plt.subplots(3, 1, sharex=True)

        sns.lineplot(ax=axes[0], x='Time', y='Position', data=odometry)
        sns.lineplot(ax=axes[1], x='Time', y='Velocity', data=odometry)
        sns.lineplot(ax=axes[2], x=odometry['Time'], y=thrust['Thrust'])

        plt.savefig(plot_dir.joinpath('1D-measurements-white.eps'))


    ''' Create latex tables to show difference in sample times '''

    if print_tables:

        table_dir = Path('tables').joinpath(RUN_NAME)

        times_head = odometry.head().join(thrust.head(), lsuffix='_odom', rsuffix='_thrust')[['Time_odom', 'Time_thrust']]

        with table_dir.joinpath('times-head.tex').open(mode='w') as f:
            print(times_head.to_latex(label='tab:times-head', caption='Timestamps of first samples'), file=f)


        df = odometry[odometry['Time'] > 41.82].head().reset_index().join(thrust.head(), lsuffix='_odom', rsuffix='_thrust')[['Time_odom', 'Time_thrust']]

        with table_dir.joinpath('uneq-sampling-freq.tex').open(mode='w') as f:
            print(df.to_latex(index=False, label='tab:uneq-sampling-freq', caption='Comparison of sampling frequencies'), file=f) 

        


if __name__ == "__main__":
    main()
