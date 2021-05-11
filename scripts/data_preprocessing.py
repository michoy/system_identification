'''
Script that prepares data for later usage

'''

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns


def double_line_plot(odometry: pd.DataFrame, thrust: pd.DataFrame, save_path: Path):
    sns.set()
    fig, axes = plt.subplots(2, 1, sharex=True)

    odometry = odometry.head(2000)
    thrust = thrust.head(2000)

    sns.lineplot(ax=axes[0], y=odometry['linear.x'], x=odometry['Time'])
    sns.lineplot(ax=axes[1], y=thrust['force.x'], x=thrust['Time'])

    plt.savefig(save_path)


def check_for_missing_values(data: pd.DataFrame) -> None:
    if data.isnull().values.any():
        print('The dataset has %d missing values' % data.isnull().sum().sum())
        for col_name in data.columns:
            print('Column %s has %d missing values' % (col_name, data[col_name].isnull().sum()))


def recreate_sampling_times(
    data: pd.DataFrame, 
    step_length: float, 
    duration: float, 
    plot_col=None
    ) -> pd.DataFrame:

    '''
    Functions that transforms measurement data with samples taken it any (possibly irregular) 
    sample rate and outputs the same measurements evenly spanced according to a given step length.

    data:           dataframe with numeric values that includes a 'Time' column
    step length:    desired time between each sample timestep
    duration:       amount of time covered by measurements in data
    plot_col:       name of column that should be plotted before and after (for vertification purposes)
    '''

    start_time = data['Time'].iloc[0]
    get_shifted_time = lambda row: row['Time'] - start_time
    shifted_timestamps = data.apply(get_shifted_time, axis=1).rename('Time', axis=1)

    columns_except_time = data.columns.difference(['Time'])
    timesteps = np.arange(0, duration, step_length)

    new_columns = [pd.Series(timesteps, name='Time')]

    for col_name in columns_except_time:
        f = interp1d(shifted_timestamps.values, data[col_name].values)
        new_columns.append(pd.Series(f(timesteps), name=col_name))

    data_new = pd.concat(new_columns, axis=1)

    if plot_col:
        SAVEDIR = Path('plots/interpolation')
        sns.set_style('white')
        plt.figure(figsize=(5, 2.5))
        sns.lineplot(x=shifted_timestamps.values, y=data[plot_col], label='original')
        sns.lineplot(x='Time', y=plot_col, data=data_new, label='interpolated')
        plt.ylabel('Velocity')
        plt.savefig(SAVEDIR.joinpath('%s.eps' % plot_col))

    return data_new


def main():

    ''' Define paths for loading and saving dataframes '''

    RUN_NAME = '1D-run-1'
    ODOM_PATH = Path('data/%s/manta-pose_gt.csv' % RUN_NAME)
    THRUST_PATH = Path('data/%s/manta-thruster_manager-input.csv' % RUN_NAME)
    SAVEDIR = Path('data/processed')
    DEBUG_PLOT_DIR = Path('plots/debug')


    ''' Load Dataframes '''

    odometry = pd.read_csv(ODOM_PATH)
    thrust = pd.read_csv(THRUST_PATH)


    ''' Check for missing values '''

    check_for_missing_values(odometry)
    check_for_missing_values(thrust)


    ''' Clean up dataframes '''

    drop_columns = ['header.seq', 'header.stamp.secs', 'header.stamp.nsecs', 
    'header.frame_id', 'child_frame_id', 'pose.covariance', 'twist.covariance']
    odometry = (odometry
        .drop(columns=drop_columns, axis='columns')     # covariance is dropped because of interpolation difficulties
        .rename(columns={
            'pose.pose.position.x': 'position.x', 
            'pose.pose.position.y': 'position.y', 
            'pose.pose.position.z': 'position.z',
            'pose.pose.orientation.x': 'orientation.x', 
            'pose.pose.orientation.y': 'orientation.y',
            'pose.pose.orientation.z': 'orientation.z', 
            'pose.pose.orientation.w': 'orientation.w', 
            'twist.twist.linear.x': 'linear.x', 
            'twist.twist.linear.y': 'linear.y', 
            'twist.twist.linear.z': 'linear.z',
            'twist.twist.angular.x': 'angular.x', 
            'twist.twist.angular.y': 'angular.y',
            'twist.twist.angular.z': 'angular.z'
    }))


    ''' Synchronize sampling times '''

    odom_init_time: float = odometry['Time'].iloc[0]
    odom_end_time: float = odometry['Time'].iloc[-1]
    thrust_init_time: float = thrust['Time'].iloc[0]
    thrust_end_time: float = thrust['Time'].iloc[-1]

    # define time boundry
    init_time = max(odom_init_time, thrust_init_time)
    end_time = min(odom_end_time, thrust_end_time)

    # only keep rows inside time boundry
    odometry = odometry[odometry['Time'] > init_time]
    odometry = odometry[odometry['Time'] < end_time]
    thrust = thrust[thrust['Time'] > init_time]
    thrust = thrust[thrust['Time'] < end_time]

    # create synchronized datasets through interpolation
    thrust = recreate_sampling_times(thrust, step_length=0.1, duration=end_time-init_time)
    odometry = recreate_sampling_times(odometry, step_length=0.1, duration=end_time-init_time, plot_col='linear.x')


    ''' Merge dataframes '''

    data = odometry.merge(thrust, how='inner', left_on='Time', right_on='Time').set_index('Time')


    ''' Save processed data '''
 
    data.to_csv(SAVEDIR.joinpath('%s.csv' % RUN_NAME))
    data.to_pickle(SAVEDIR.joinpath('%s.pickle' % RUN_NAME))


if __name__ == "__main__":
    main()
