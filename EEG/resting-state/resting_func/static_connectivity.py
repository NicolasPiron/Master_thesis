import os
import mne
from mne_connectivity import spectral_connectivity_time
from mne_connectivity.viz import plot_connectivity_circle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_conn_matrix_subject(subject_id, metric, freqs, input_dir, output_dir):
    ''' Create connectivity matrix for a single subject, for a single metric and frequency band. 
    Plot the connectivity matrix and save it as 2 png files (circle and matrix). Save the connectivity matrix as a csv file.
    
    Parameters
    ----------
    subject_id : str
        Subject ID
    metric : str
        Connectivity metric (i.e. plv or pli).
    freqs : list
        List of frequencies to be used for connectivity analysis.
    input_dir : str
        Path to input directory.
    output_dir : str
        Path to output directory.

    Returns
    -------
    df_open : pandas dataframe
        Connectivity matrix for eyes open condition.
    df_closed : pandas dataframe
        Connectivity matrix for eyes closed condition.
    '''

    # Load resting state data (eyes open and eyes closed)
    rs_open_epochs_path = os.path.join(input_dir, f'sub-{subject_id}', 'RESTINGSTATEOPEN', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-RESTINGSTATEOPEN.fif')
    rs_closed_epochs_path = os.path.join(input_dir, f'sub-{subject_id}', 'RESTINGSTATECLOSE', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-RESTINGSTATECLOSE.fif')

    try:
        rs_open_epochs = mne.read_epochs(rs_open_epochs_path)
    except:
        print(f'No file found for subject {subject_id} in {rs_open_epochs_path}')
    try:
        rs_closed_epochs = mne.read_epochs(rs_closed_epochs_path)
    except:
        print(f'No file found for subject {subject_id} in {rs_closed_epochs_path}')


    rs_open_epochs.info['bads'] = []
    rs_closed_epochs.info['bads'] = []
    print('===== Bad channels reset =====')
    rs_open_epochs.pick_types(eeg=True)
    rs_closed_epochs.pick_types(eeg=True)
    print('===== EEG channels selected =====')

    chan_names = rs_open_epochs.ch_names

    # Create connectivity matrix
    rs_open_conn_2D = spectral_connectivity_time(rs_open_epochs, freqs=freqs, method=metric, sfreq=rs_open_epochs.info['sfreq'], fmin=freqs[0], fmax=freqs[-1], average=True, faverage=True, mode='multitaper').get_data().reshape(64, 64)
    rs_closed_conn_2D = spectral_connectivity_time(rs_closed_epochs, freqs=freqs, method=metric, sfreq=rs_closed_epochs.info['sfreq'], fmin=freqs[0], fmax=freqs[-1],average=True, faverage=True, mode='multitaper').get_data().reshape(64, 64)
    print(f'===== Connectivity matrices created for subject {subject_id} =====')

    # Plot connectivity matrix, 1 plot for each condition. Add title with subject ID, condition and frequency band.
    # Also add a colorbar to each plot.
    def plot_conn_matrix(conn_matrix):

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f'Subject {subject_id} - {metric} - {freqs[0]}-{freqs[-1]} Hz')
        ax.set_xticks(np.arange(len(chan_names)))
        ax.set_yticks(np.arange(len(chan_names)))
        ax.set_xticklabels(chan_names, rotation=90, fontsize=8)
        ax.set_yticklabels(chan_names, fontsize=8)
        im0 = ax.imshow(conn_matrix)
        fig.colorbar(im0, shrink=0.81)

        return fig
    
    def plot_conn_circle(conn_matrix):

        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black',
                       subplot_kw=dict(polar=True))
        plot_connectivity_circle(conn_matrix, node_names=chan_names,n_lines=300,
                                title=f'Subject {subject_id} - {metric} - {freqs[0]}-{freqs[-1]} Hz', ax=ax, show=False)
        fig.tight_layout()
    
        return fig

    fig_open = plot_conn_matrix(rs_open_conn_2D)
    plt.close()
    fig_closed = plot_conn_matrix(rs_closed_conn_2D)
    plt.close()
    fig_open_circle = plot_conn_circle(rs_open_conn_2D) 
    fig_closed_circle = plot_conn_circle(rs_closed_conn_2D)
    print(f'===== Connectivity plots created for subject {subject_id} =====')

    # Save figures
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATEOPEN', 'connectivity', 'static', 'figs')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATEOPEN', 'connectivity', 'static', 'figs'))
    fig_open.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATEOPEN', 'connectivity', 'static', 'figs', f'sub-{subject_id}-static-{metric}-{freqs[0]}-{freqs[-1]}-open.png'))
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATECLOSE', 'connectivity', 'static', 'figs')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATECLOSE', 'connectivity', 'static', 'figs'))
    fig_closed.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATECLOSE', 'connectivity', 'static', 'figs', f'sub-{subject_id}-static-{metric}-{freqs[0]}-{freqs[-1]}-closed.png'))
    fig_open_circle.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATEOPEN', 'connectivity', 'static', 'figs', f'sub-{subject_id}-static-{metric}-{freqs[0]}-{freqs[-1]}-open-circle.png'))
    fig_closed_circle.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATECLOSE', 'connectivity', 'static', 'figs', f'sub-{subject_id}-static-{metric}-{freqs[0]}-{freqs[-1]}-closed-circle.png'))
    print(f'===== Connectivity plots saved for subject {subject_id} =====')

    # Save connectivity matrix as csv
    df_open = pd.DataFrame(rs_open_conn_2D, columns=chan_names, index=chan_names)
    df_closed = pd.DataFrame(rs_closed_conn_2D, columns=chan_names, index=chan_names)
    
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATEOPEN', 'connectivity', 'static', 'conn_data')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATEOPEN', 'connectivity', 'static', 'conn_data'))
    df_open.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATEOPEN', 'connectivity', 'static', 'conn_data', f'sub-{subject_id}-static-{metric}-{freqs[0]}-{freqs[-1]}-open.csv'))
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATECLOSE', 'connectivity', 'static', 'conn_data')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATECLOSE', 'connectivity', 'static', 'conn_data'))
    df_closed.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATECLOSE', 'connectivity', 'static', 'conn_data', f'sub-{subject_id}-static-{metric}-{freqs[0]}-{freqs[-1]}-closed.csv'))
    print(f'===== Connectivity matrices saved for subject {subject_id} =====')

    return df_open, df_closed

def create_conn_matrix_group(subject_list, metric, freqs, input_dir, output_dir):
    ''' Create connectivity matrices for a group of subjects, for a single metric and frequency band (2 conditions - eyes open and closed).


    Parameters
    ----------
    subject_list : list
        List of subject IDs.
    metric : str
        Connectivity metric (i.e. plv or pli).
    freqs : list
        List of frequencies to be used for connectivity analysis.
    input_dir : str
        Path to input directory.
    output_dir : str
        Path to output directory.

    Returns
    -------
    df_open_group : pandas dataframe
        Group average connectivity matrix for eyes open condition.
    df_closed_group : pandas dataframe
        Group average connectivity matrix for eyes closed condition.
    '''

    matrices_open = []
    matrices_closed = []

    subject_data_not_found = []

    for subject_id in subject_list:

        subject_id = str(subject_id).zfill(2)

        try:
            if os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATEOPEN', 'connectivity', 'static', 'conn_data', f'sub-{subject_id}-static-{metric}-{freqs[0]}-{freqs[-1]}-open.csv')) and os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATECLOSE', 'connectivity', 'static', 'conn_data', f'sub-{subject_id}-static-{metric}-{freqs[0]}-{freqs[-1]}-closed.csv')):
                print(f'===== Loading connectivity matrices for subject {subject_id} =====')
                df_open = pd.read_csv(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATEOPEN', 'connectivity', 'static', 'conn_data', f'sub-{subject_id}-static-{metric}-{freqs[0]}-{freqs[-1]}-open.csv'), index_col=0)
                df_closed = pd.read_csv(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATECLOSE', 'connectivity', 'static', 'conn_data', f'sub-{subject_id}-static-{metric}-{freqs[0]}-{freqs[-1]}-closed.csv'), index_col=0)
            else:
                print(f'===== Connectivity matrices not found for {subject_id}, creating it now =====')
                df_open, df_closed = create_conn_matrix_subject(subject_id, metric, freqs, input_dir, output_dir)

            matrices_open.append(df_open)
            matrices_closed.append(df_closed)
            print(f'===== Connectivity matrices for subject {subject_id} appended to list =====')
        except:
            print(f'===== Error loading connectivity matrices for subject {subject_id} =====')
            subject_data_not_found.append(subject_id)
            continue

    # Create group average connectivity matrix
    df_open_group = pd.concat(matrices_open, axis=0).groupby(level=0).mean()
    df_closed_group = pd.concat(matrices_closed, axis=0).groupby(level=0).mean()
    print(f'===== Group average connectivity matrices created. Subjects {subject_data_not_found} were not included =====')

    return df_open_group, df_closed_group


def plot_and_save_group_matrix(df_open_group, df_closed_group, population, metric, freqs, output_dir):
    ''' Plot and save group average connectivity matrices for a single metric and frequency band (2 conditions - eyes open and closed).

    Parameters
    ----------
    df_open_group : pandas dataframe
        Group average connectivity matrix for eyes open condition.
    df_closed_group : pandas dataframe
        Group average connectivity matrix for eyes closed condition.
    population : str
        Population name (based on the subject list given in create_conn_matrix_group)
    metric : str
        Connectivity metric (i.e. plv or pli).
    freqs : list
        List of frequencies to be used for connectivity analysis.
    output_dir : str
        Path to output directory.

    Returns
    -------
    None.
    '''


    # get the channel names
    chan_names = df_open_group.index.values

    # Plot group average connectivity matrix, 1 plot for each condition. Add title with subject ID, condition and frequency band.
    # Also add a colorbar to each plot.
    def plot_conn_matrix(conn_matrix, condition):

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f'{population} - {condition} - {metric} - {freqs[0]}-{freqs[-1]} Hz')
        ax.set_xticks(np.arange(len(chan_names)))
        ax.set_yticks(np.arange(len(chan_names)))
        ax.set_xticklabels(chan_names, rotation=90, fontsize=8)
        ax.set_yticklabels(chan_names, fontsize=8)
        im0 = ax.imshow(conn_matrix)
        fig.colorbar(im0, shrink=0.81)

        return fig
    
    def plot_conn_circle(conn_matrix, condition):

        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black',
                       subplot_kw=dict(polar=True))
        plot_connectivity_circle(conn_matrix, node_names=chan_names,n_lines=300,
                                title=f'{population} - {condition} - {metric} - {freqs[0]}-{freqs[-1]} Hz', ax=ax, show=False)
        fig.tight_layout()
    
        return fig

    fig_open = plot_conn_matrix(df_open_group, 'open')
    plt.close()
    fig_closed = plot_conn_matrix(df_closed_group, 'closed')
    plt.close()
    fig_open_circle = plot_conn_circle(df_open_group, 'open') 
    fig_closed_circle = plot_conn_circle(df_closed_group, 'closed')
    print(f'===== Connectivity plots created for {population} =====')

    # Save figures
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', population, 'figs')):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', population, 'figs'))
    fig_open.savefig(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', population, 'figs', f'{population}-static-{metric}-{freqs[0]}-{freqs[-1]}-open.png'))
    fig_open_circle.savefig(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', population, 'figs', f'{population}-static-{metric}-{freqs[0]}-{freqs[-1]}-open-circle.png'))
    fig_closed.savefig(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', population, 'figs', f'{population}-static-{metric}-{freqs[0]}-{freqs[-1]}-closed.png'))
    fig_closed_circle.savefig(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', population, 'figs', f'{population}-static-{metric}-{freqs[0]}-{freqs[-1]}-closed-circle.png'))
    print(f'===== Connectivity plots saved for {population} =====')

    # Save connectivity matrix as csv
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', population, 'conn_data')):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', population, 'conn_data'))
    df_open_group.to_csv(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', population, 'conn_data', f'{population}-static-{metric}-{freqs[0]}-{freqs[-1]}-open.csv'))
    df_closed_group.to_csv(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', population, 'conn_data', f'{population}-static-{metric}-{freqs[0]}-{freqs[-1]}-closed.csv'))
    print(f'===== Connectivity matrices saved for {population} =====')