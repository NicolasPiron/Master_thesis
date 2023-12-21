import os
import re
import mne
from mne_connectivity import spectral_connectivity_time
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colour import Color

def plot_conn_matrix(conn_matrix, population, metric, freqs, condition):
    '''
    Plot connectivity matrix for a single population, metric, frequency band and condition.
    Only to plot poplutation average connectivity matrix.

    Parameters
    ----------
    conn_matrix : pandas dataframe
        Connectivity matrix.
    population : str
        Population name (based on the subject list given in create_conn_matrix_group)
    metric : str
        Connectivity metric (i.e. plv or pli).
    freqs : list
        List of frequencies to be used for connectivity analysis.
    condition : str
        Condition name (i.e. open or closed).
    
    Returns
    -------
    fig : matplotlib figure
        Connectivity matrix.
    '''

    chan_names = conn_matrix.index.values
    conn_matrix = conn_matrix.values

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f'{population} - {condition} - {metric} - {freqs[0]}-{freqs[-1]} Hz')
    ax.set_xticks(np.arange(len(chan_names)))
    ax.set_yticks(np.arange(len(chan_names)))
    ax.set_xticklabels(chan_names, rotation=90, fontsize=8)
    ax.set_yticklabels(chan_names, fontsize=8)
    im0 = ax.imshow(conn_matrix, vmin=0, vmax=1)
    fig.colorbar(im0,shrink=0.81)

    return fig
    
def plot_conn_circle(conn_matrix, population, metric, freqs, condition):
    '''
    Plot connectivity circle for a single population, metric, frequency band and condition.

    Parameters
    ----------
    conn_matrix : pandas dataframe
        Connectivity matrix.
    population : str
        Population name (based on the subject list given in create_conn_matrix_group)
    metric : str
        Connectivity metric (i.e. plv or pli).
    freqs : list
        List of frequencies to be used for connectivity analysis.
    condition : str
        Condition name (i.e. open or closed).

    Returns
    -------
    fig : matplotlib figure
        Connectivity circle.
    '''

    # Create a gradient
    red = Color("red")
    colors = list(red.range_to(Color("blue"),64))
    color_list = [col.get_rgb() for col in colors]

    # Reorder channels
    new_node_order= ['Fpz', 'AFz', 'Fz', 'FCz', 'Cz','Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3','FC1', 'C1',
                    'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3','O1', 'Iz', 'Oz', 'POz',
                    'Pz', 'CPz', 'O2', 'PO4', 'PO8', 'P10', 'P8', 'P6', 'P4', 'P2', 'CP2', 'CP4',
                    'CP6', 'TP8', 'T8', 'C6', 'C4', 'C2', 'FC2', 'FC4', 'FC6', 'FT8', 'F8', 'F6', 'F4', 'F2', 'AF4', 'AF8','Fp2']

    # Reorder colors
    chan_names = conn_matrix.index.values
    index_list=[]
    for ch_name in chan_names:
        new_idx = new_node_order.index(ch_name)
        index_list.append(new_idx)
    correct_order = [color_list[i] for i in index_list]

    # Create node angles
    node_angles = circular_layout(chan_names, new_node_order, start_pos=74,
                                group_boundaries=[0, 5, 32, 37], group_sep=3)

    conn_matrix = conn_matrix.values

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black',
                    subplot_kw=dict(polar=True))
    plot_connectivity_circle(conn_matrix, node_names=chan_names, node_angles=node_angles, node_colors=correct_order,
                            vmin=0.6, vmax=1, n_lines=300,
                            title= f'{population} - {condition} - {metric} - {freqs[0]}-{freqs[-1]} Hz' ,
                            ax=ax, show=False)


    return fig

def create_conn_matrix_subject(subject_id, metric, freqs, input_dir, output_dir):
    ''' 
    Create connectivity matrix for a single subject, for a single metric and frequency band. 
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

    # Get channel names, important for plotting
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
        im0 = ax.imshow(conn_matrix, vmin=0, vmax=1)
        fig.colorbar(im0, shrink=0.81)

        return fig
    
    def plot_conn_circle(conn_matrix):

    # Create a gradient
        red = Color("red")
        colors = list(red.range_to(Color("blue"),64))
        color_list = [col.get_rgb() for col in colors]

        # Reorder channels
        new_node_order= ['Fpz', 'AFz', 'Fz', 'FCz', 'Cz','Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3','FC1', 'C1',
                        'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3','O1', 'Iz', 'Oz', 'POz',
                        'Pz', 'CPz', 'O2', 'PO4', 'PO8', 'P10', 'P8', 'P6', 'P4', 'P2', 'CP2', 'CP4',
                        'CP6', 'TP8', 'T8', 'C6', 'C4', 'C2', 'FC2', 'FC4', 'FC6', 'FT8', 'F8', 'F6', 'F4', 'F2', 'AF4', 'AF8','Fp2']

        # Reorder colors
        index_list=[]
        for ch_name in chan_names:
            new_idx = new_node_order.index(ch_name)
            index_list.append(new_idx)
        correct_order = [color_list[i] for i in index_list]

        # Create node angles
        node_angles = circular_layout(chan_names, new_node_order, start_pos=74,
                                    group_boundaries=[0, 5, 32, 37], group_sep=3)

        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black',
                        subplot_kw=dict(polar=True))
        plot_connectivity_circle(conn_matrix, node_names=chan_names, node_angles=node_angles, node_colors=correct_order,
                                vmin=0.6, vmax=1, n_lines=300,
                                title= f'Subject {subject_id} - {metric} - {freqs[0]}-{freqs[-1]} Hz' ,
                                ax=ax, show=False)
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
    ''' 
    Create connectivity matrices for a group of subjects, for a single metric and frequency band (2 conditions - eyes open and closed).

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
    df_open_group = 0
    df_closed_group = 0
    for i in range(len(matrices_open)):
        if i == 0:
            df_open_group = matrices_open[i].copy()
            df_closed_group = matrices_closed[i].copy()
        else:
            df_open_group += matrices_open[i].copy()
            df_closed_group += matrices_closed[i].copy()
    df_open_group = df_open_group/len(matrices_open)
    df_closed_group = df_closed_group/len(matrices_closed)

    print(f'===== Group average connectivity matrices created. Subjects {subject_data_not_found} were not included =====')

    return df_open_group, df_closed_group

def plot_and_save_group_matrix(df_open_group, df_closed_group, population, metric, freqs, output_dir):
    ''' 
    Plot and save group average connectivity matrices for a single metric and frequency band (2 conditions - eyes open and closed).

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

    fig_open = plot_conn_matrix(df_open_group, population, metric, freqs, 'open')
    plt.close()
    fig_closed = plot_conn_matrix(df_closed_group, population, metric, freqs, 'closed')
    plt.close()
    fig_open_circle = plot_conn_circle(df_open_group, population, metric, freqs, 'open') 
    fig_closed_circle = plot_conn_circle(df_closed_group, population, metric, freqs, 'closed')
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

def find_elements_in_dir_name(dir_name):
    '''
    Find the different elements in a directory name. The directory name should be formatted as follows:
    group1-frequency1-condition1_VS_group2-frequency2-condition2

    Parameters
    ----------
    dir_name : str
        Directory name.
    
    Returns
    -------
    grp1 : str
        Group 1 name.
    grp2 : str
        Group 2 name.
    freq1 : str
        Frequency 1 name.
    freq2 : str
        Frequency 2 name.
    cond1 : str
        Condition 1 name.
    cond2 : str
        Condition 2 name.
    '''

    element_list = re.split('[_-]', dir_name)

    name_pattern = re.compile(r'(old|young|stroke|pulvinar|thal)')
    frequency_pattern = re.compile(r'(theta|alpha|low|high)')
    condition_pattern = re.compile(r'(open|closed)')

    groups = []
    frequencies = []
    conditions = []

    for element in element_list:
        if name_pattern.search(element):
            groups.append(name_pattern.search(element).group())
        elif frequency_pattern.search(element):
            frequencies.append(frequency_pattern.search(element).group())
        elif condition_pattern.search(element):
            conditions.append(condition_pattern.search(element).group())

    grp1 = groups[0]
    grp2 = groups[-1]
    freq1 = frequencies[0]
    freq2 = frequencies[-1]
    cond1 = conditions[0]
    cond2 = conditions[-1]

    return grp1, grp2, freq1, freq2, cond1, cond2

def load_conn_mat(input_dir, group, freq, cond):
    '''
    Load connectivity matrix for a specific group, frequency and condition.

    Parameters
    ----------
    group : str
        Group name.
    freq : str
        Frequency name.
    cond : str
        Condition name.

    Returns
    -------
    df : pandas dataframe
        Connectivity matrix.
    '''
    if group == 'old':
        full_group = 'old_control'
    elif group == 'young':
        full_group = 'young_control'
    elif group == 'thal':
        full_group = 'thal_control'
    elif group == 'stroke':
        full_group = 'stroke'
    elif group == 'pulvinar':
        full_group = 'pulvinar'

    if freq == 'theta':
        freq = [4, 8]
    elif freq == 'alpha':
        freq = [8, 12]
    elif freq == 'low':
        freq = [12, 16]
    elif freq == 'high':
        freq = [16, 30]

    print(f'Loading connectivity matrix for {group} - {freq} - {cond}')
    df = pd.read_csv(os.path.join(input_dir, 'all_subj', 'resting-state', 'connectivity', 'static', full_group, 'conn_data', f'{full_group}-static-plv-{freq[0]}-{freq[1]}-{cond}.csv'), index_col=0)
    print(f'Connectivity matrix for {group} - {freq} - {cond} loaded')

    return df

def multiply_by_adjacency_matrix(df, adjacency_matrix):
    '''
    Multiply connectivity matrix by adjacency matrix to keep only the significant connexions.

    Parameters
    ----------
    df : pandas dataframe
        Connectivity matrix.
    adjacency_matrix : pandas dataframe
        Adjacency matrix.
    
    Returns
    -------
    df : pandas dataframe
        Connectivity matrix with only significant connexions.
    '''
    df = df.copy()
    print('Multiplying connectivity matrix by adjacency matrix')
    for i in range(len(df)):
        for j in range(len(df)):
            df.iloc[i, j] = df.iloc[i, j] * adjacency_matrix.iloc[i, j]
    print('Connectivity matrix multiplied by adjacency matrix')
    return df
    
def create_significant_conn_mat(input_dir, output_dir):

    # List all directories in nbs_results
    nbs_results = os.listdir(os.path.join(input_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'nbs_results'))
    nbs_results.remove('all_pvals')
    try:
        nbs_results.remove('.DS_Store')
    except:
        pass


    for dir in nbs_results:

        try:
            # Load adjacency matrix
            adjacency_matrix = pd.read_csv(os.path.join(input_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'nbs_results', dir, 'adj.csv'), index_col=0)
            print(f'Adjacency matrix loaded for {dir}')
            # Make sure adjacency matrix only contains 0 and 1
            adjacency_matrix = adjacency_matrix.applymap(lambda x: 1 if x > 1 else x)
            # Load connectivity matrices
            grp1, grp2, freq1, freq2, cond1, cond2 = find_elements_in_dir_name(dir)

            df_1 = load_conn_mat(input_dir, grp1, freq1, cond1)
            df_2 = load_conn_mat(input_dir, grp2, freq2, cond2)

            sign_df_1 = multiply_by_adjacency_matrix(df_1, adjacency_matrix)
            sign_df_2 = multiply_by_adjacency_matrix(df_2, adjacency_matrix)

            name_1 = f'{grp1}-{freq1}-{cond1}'
            name_2 = f'{grp2}-{freq2}-{cond2}'
            sign_df_1.to_csv(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'nbs_results', dir, f'{name_1}-sign.csv'))
            sign_df_2.to_csv(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'nbs_results', dir, f'{name_2}-sign.csv'))
            print(f'Significant connectivity matrices saved for {dir}')
        except:
            print(f'Error with {dir}')
            continue

def plot_significant_conn_mat(input_dir, output_dir):

    # List all directories in nbs_results
    nbs_results = os.listdir(os.path.join(input_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'nbs_results'))
    nbs_results.remove('all_pvals')
    try:
        nbs_results.remove('.DS_Store')
    except:
        pass

    for dir in nbs_results:
       
        try:

            grp1, grp2, freq1, freq2, cond1, cond2 = find_elements_in_dir_name(dir)
            name_1 = f'{grp1}-{freq1}-{cond1}'
            name_2 = f'{grp2}-{freq2}-{cond2}'

            if freq1 == 'theta':
                freq1 = [4, 8]
            elif freq1 == 'alpha':
                freq1 = [8, 12]
            elif freq1 == 'low':
                freq1 = [12, 16]
            elif freq1 == 'high':
                freq1 = [16, 30]

            if freq2 == 'theta':
                freq2 = [4, 8]
            elif freq2 == 'alpha':
                freq2 = [8, 12]
            elif freq2 == 'low':
                freq2 = [12, 16]
            elif freq2 == 'high':
                freq2 = [16, 30]
        
            sign_df_1 = pd.read_csv(os.path.join(input_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'nbs_results', dir, f'{name_1}-sign.csv'), index_col=0)
            sign_df_2 = pd.read_csv(os.path.join(input_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'nbs_results', dir, f'{name_2}-sign.csv'), index_col=0)
            print(f'Significant connectivity matrices loaded for {dir}')

            mat_fig_1 = plot_conn_matrix(sign_df_1, grp1, 'plv', freq1, cond1)
            plt.close()
            mat_fig_2 = plot_conn_matrix(sign_df_2, grp2, 'plv', freq2, cond2)
            plt.close()
            print(f'Connectivity plots created for {dir}')
            mat_fig_1.savefig(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'nbs_results', dir, f'{name_1}-sign.png'))
            mat_fig_2.savefig(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'nbs_results', dir, f'{name_2}-sign.png'))
            print(f'Connectivity plots saved for {dir}')
            
            circle_fig_1 = plot_conn_circle(sign_df_1, grp1, 'plv', freq1, cond1)
            plt.close()
            circle_fig_2 = plot_conn_circle(sign_df_2, grp2, 'plv', freq2, cond2)
            plt.close()
            print(f'Connectivity circles created for {dir}')
            circle_fig_1.savefig(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'nbs_results', dir, f'{name_1}-sign-circle.png'), dpi=300)
            circle_fig_2.savefig(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'nbs_results', dir, f'{name_2}-sign-circle.png'), dpi=300)
            print(f'Connectivity circles saved for {dir}')
        except:
            print(f'Error with {dir}')
            continue