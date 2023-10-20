import mne
import os
import pandas as pd
import glob
import numpy as np

### ================================================================================================
### =============================== RESTING STATE ALPHA - THETA POWER ==============================
### ================================================================================================

def get_resting_power(subject_id, input_dir, condition, freqs, right_elecs, left_elecs):
    ''' Computes the power of a certain freq band for a given condition and returns the mean power for each side. 

    Parameters
    ----------
    subject_id : str
        The subject ID.
    input_dir : str
        The path to the directory containing all the subject directories
    condition : str
        The condition to compute the power for.
    freqs : list of int
        The frequencies to compute the power for.
    right_elecs : list of str
        The electrodes to compute the power for on the right side.
    left_elecs : list of str
        The electrodes to compute the power for on the left side.

    Returns
    -------
    right_power_mean : float
        The mean power for the right side.
    left_power_mean : float
        The mean power for the left side.

    '''
    # We could use the whole epoch set or an evoked object to save some time. For now I'm trying with the whole epoch set. 

    # Load the epochs
    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-{condition}.fif'))
   
    # Compute the power
    right_power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=freqs / 2., picks=right_elecs,
                                            use_fft=True, return_itc=False, decim=1,
                                            n_jobs=1, verbose=True)

    left_power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=freqs / 2., picks=left_elecs,
                                            use_fft=True, return_itc=False, decim=1,
                                            n_jobs=1, verbose=True)

    return right_power, left_power


def get_resting_power_df(input_dir, output_dir):
    ''' Create a df with the resting state power for each subject and each resting state condition,
    with different electrodes and frequencies (alpha and theta bands).
    
    Parameters
    ----------
    input_dir : str
        The path to the directory containing all the subject directories
    output_dir : str
        The path to the directory where the dataframe will be saved

    Returns
    ----------
    df : pandas.DataFrame
        A dataframe with the resting state power for each subject and each resting state condition,
        with different elctrodes and frequencies (alpha and theta bands).
    '''

    # Define the conditions
    conditions = ['RESTINGSTATEOPEN', 'RESTINGSTATECLOSE']

    # Define the frequencies
    alpha_freqs = np.arange(8, 13)
    theta_freqs = np.arange(4, 9)
    bands = [alpha_freqs, theta_freqs]

    # Define the electrodes
    cluster_dict = {'occipital': {'right':['O2', 'PO4', 'PO8'], 'left': ['O1', 'PO3', 'PO7']},
                    'parietal' : {'right':['P2', 'CP2', 'CP4'], 'left':['P1', 'CP1', 'CP3']},
                    'frontal':{'right':['FC2', 'FC4', 'F2'], 'left':['FC1', 'FC3', 'F1']},
                    'total':{ 'right':['O2', 'PO4', 'PO8', 'P2', 'CP2', 'CP4', 'FC2', 'FC4', 'F2'],
                             'left':['O1', 'PO3', 'PO7', 'P1', 'CP1', 'CP3', 'FC1', 'FC3', 'F1']}}
    
    # Define the subject list
    subject_list = [os.path.basename(subj) for subj in glob.glob(os.path.join(input_dir, 'sub-*'))]

    # Initiate the df
    df = pd.DataFrame(columns=[['ID', 'eyes', 'freq band', 'cluster', 'side', 'mean power']])

    # Create a counter to keep track of the rows and be able to correctly add the data to the df (bad idea, need a better solution)
    counter = 0

    # Loop over the subjects
    for subject in subject_list:

        # Check if the files exist, if not, skip the subject
        if not os.path.exists(os.path.join(input_dir, subject, 'RESTINGSTATEOPEN', 'cleaned_epochs', f'{subject}-cleaned_epochs-RESTINGSTATEOPEN.fif')) or not os.path.exists(os.path.join(input_dir, subject, 'RESTINGSTATECLOSE', 'cleaned_epochs', f'{subject}-cleaned_epochs-RESTINGSTATECLOSE.fif')):
            print(f'========================= No resting state epochs found for {subject}')
            continue

        subject = subject[-2:]
        print(f'========================= working on {subject}')
        # Loop over the frequencies
        for freq in bands:
            print(f'========================= working on {freq}')
            # Loop over the clusters
            for cluster in cluster_dict.keys():
                print(f'========================= working on {cluster}')
                # Loop over the conditions
                for condition in conditions:
                    print(f'========================= working on {condition}')
                    # Compute the power
                    right_power, left_power = get_resting_power(subject_id=subject, input_dir=input_dir, condition=condition, freqs=freq, right_elecs=cluster_dict[cluster]['right'], left_elecs=cluster_dict[cluster]['left'])

                    # Compute the mean power
                    right_power_mean = right_power.to_data_frame()[cluster_dict[cluster]['right']].mean(axis=1).mean(axis=0)
                    left_power_mean = left_power.to_data_frame()[cluster_dict[cluster]['left']].mean(axis=1).mean(axis=0)

                    # Give better names to the values - better readability
                    if freq.mean() == 10:
                        freq_ = 'alpha'
                    elif freq.mean() == 6:
                        freq_ = 'theta'
                    
                    if condition == 'RESTINGSTATEOPEN':
                        condition_ = 'open'
                    elif condition == 'RESTINGSTATECLOSE':
                        condition_ = 'closed'

                    # Add the data to the df
                    df.loc[counter] = [subject, condition_, freq_, cluster, 'right', right_power_mean]
                    # Adjust the counter so that the next row is added correctly
                    counter += 1
                    df.loc[counter] = [subject, condition_, freq_, cluster, 'left', left_power_mean]
                    counter += 1
                    
    # Scientific notification because very small values
    pd.options.display.float_format = '{:.5e}'.format

    # Save the df
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'resting-state', 'resting-state-power-df')):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'resting-state','resting-state-power-df'))
    df.to_csv(os.path.join(output_dir, 'all_subj', 'resting-state', 'resting-state-power-df', 'allsubj_resting_state_power.csv'))

    print('========================= DONE :D')

    return df


### ================================================================================================
### =============================== USING SPECTRUM OBJECTS =========================================
### ================================================================================================

def get_spectrum(subject_id, input_dir):
    '''
    '''

    subject_id = str(subject_id)

    # Load the epochs
    epochs_open = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'RESTINGSTATEOPEN', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-RESTINGSTATEOPEN.fif'))
    epochs_closed = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'RESTINGSTATECLOSE', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-RESTINGSTATECLOSE.fif'))

    # Remove bad channels from info
    epochs_open.info['bads'] = []
    epochs_closed.info['bads'] = []

    # Transform the epochs into Spectrum objects
    spectrum_open = epochs_open.compute_psd(fmin=0, fmax=30)
    spectrum_closed = epochs_closed.compute_psd(fmin=0, fmax=30)

    return spectrum_open, spectrum_closed

def get_mean_freq(subject_id, input_dir, bands, picks):
    '''
    '''

    spectrum_open, spectrum_closed = get_spectrum(subject_id, input_dir)

    # Get the equivalent indices of the bands in the data (from .get_data() method)
    lower_bound = bands[0]*2
    upper_bound = bands[1]*2+1

    # Compute the mean of the epochs for each band
    epoch_average_open = spectrum_open.get_data(picks=picks)[:, :, lower_bound:upper_bound].mean(axis=0)
    epoch_average_close = spectrum_closed.get_data(picks=picks)[:, :, lower_bound:upper_bound].mean(axis=0)

    # Get the mean power for all the selected channels
    channel_average_open = epoch_average_open.mean(axis=0)
    channel_average_closed = epoch_average_close.mean(axis=0)

    # Get the mean power for the whole spectrum
    mean_power_open = np.mean(channel_average_open)
    mean_power_closed = np.mean(channel_average_closed)

    return mean_power_open, mean_power_closed

def get_spectral_df(input_dir, output_dir):

    # Define the subject list
    subject_list = [os.path.basename(subj) for subj in glob.glob(os.path.join(input_dir, 'sub-*'))]

    # Define the frequencies
    alpha_freqs = np.arange(8, 13)
    theta_freqs = np.arange(4, 9)
    bands = [alpha_freqs, theta_freqs]

    # Define the electrodes
    cluster_dict = {'occipital': {'right':['O2', 'PO4', 'PO8'], 'left': ['O1', 'PO3', 'PO7']},
                    'parietal' : {'right':['P2', 'CP2', 'CP4'], 'left':['P1', 'CP1', 'CP3']},
                    'frontal':{'right':['FC2', 'FC4', 'F2'], 'left':['FC1', 'FC3', 'F1']},
                    'total':{ 'right':['O2', 'PO4', 'PO8', 'P2', 'CP2', 'CP4', 'FC2', 'FC4', 'F2'],
                             'left':['O1', 'PO3', 'PO7', 'P1', 'CP1', 'CP3', 'FC1', 'FC3', 'F1']}}

    # Initiate the df
    df = pd.DataFrame(columns=[['ID', 'eyes', 'freq band', 'cluster', 'side', 'mean power']])

    # Create a counter to keep track of the rows and be able to correctly add the data to the df (bad idea, need a better solution)
    counter = 0

    # Loop over the subjects
    for subject in subject_list:
            
            # Check if the files exist, if not, skip the subject
            if not os.path.exists(os.path.join(input_dir, subject, 'RESTINGSTATEOPEN', 'cleaned_epochs', f'{subject}-cleaned_epochs-RESTINGSTATEOPEN.fif')) or not os.path.exists(os.path.join(input_dir, subject, 'RESTINGSTATECLOSE', 'cleaned_epochs', f'{subject}-cleaned_epochs-RESTINGSTATECLOSE.fif')):
                print(f'========================= No resting state epochs found for {subject}')
                continue
    
            subject = subject[-2:]
            print(f'========================= working on {subject}')
            # Loop over the frequencies
            for freq in bands:
                print(f'========================= working on {freq}')
                # Loop over the clusters
                for cluster in cluster_dict.keys():
                    print(f'========================= working on {cluster}')
                    # Loop over the conditions
                    
                    # Compute the mean power for right side, both conditions
                    mean_power_open_right, mean_power_closed_right = get_mean_freq(subject_id=subject, input_dir=input_dir, bands=freq, picks=cluster_dict[cluster]['right'])

                    # Compute the mean power for left side, both conditions
                    mean_power_open_left, mean_power_closed_left = get_mean_freq(subject_id=subject, input_dir=input_dir, bands=freq, picks=cluster_dict[cluster]['left'])

                    # Give better names to the values - better readability
                    if freq.mean() == 10:
                        freq_ = 'alpha'
                    elif freq.mean() == 6:
                        freq_ = 'theta'

                    # Add the data to the df
                    df.loc[counter] = [subject, 'open', freq_, cluster, 'right', mean_power_open_right]
                    # Adjust the counter so that the next row is added correctly
                    counter += 1
                    df.loc[counter] = [subject, 'closed', freq_, cluster, 'right', mean_power_closed_right]
                    counter += 1
                    df.loc[counter] = [subject, 'open', freq_, cluster, 'left', mean_power_open_left]
                    counter += 1
                    df.loc[counter] = [subject, 'closed', freq_, cluster, 'left', mean_power_closed_left]
                    counter += 1

    # Scientific notification because very small values
    pd.options.display.float_format = '{:.5e}'.format

    # Save the df
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'resting-state', 'resting-state-power-df')):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'resting-state','resting-state-power-df'))
    df.to_csv(os.path.join(output_dir, 'all_subj', 'resting-state', 'resting-state-power-df', 'allsubj_resting_state_power_spectral_version.csv'))

    print('========================= DONE :D')

    return df

def plot_resting_spectral_topo(subject_id, input_dir, output_dir, bands):

    # Get the spectrum objects
    spectrum_open, spectrum_closed = get_spectrum(subject_id, input_dir)

    # Plot the topographies
    topo_plot_open = spectrum_open.plot_topomap(ch_type='eeg', bands=bands, show=False)
    topo_plot_closed = spectrum_closed.plot_topomap(ch_type='eeg', bands=bands, show=False)

    # Save the plots
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATEOPEN', 'topo-plots')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATEOPEN', 'topo-plots'))
    topo_plot_open.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATEOPEN', 'topo-plots', f'sub-{subject_id}-topo-plot-open.png'))
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATECLOSE', 'topo-plots')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATECLOSE', 'topo-plots'))
    topo_plot_closed.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATECLOSE', 'topo-plots', f'sub-{subject_id}-topo-plot-closed.png'))

    return topo_plot_open, topo_plot_closed
    
def plot_resting_psd(subject_id, input_dir, output_dir, picks):

    # Get the spectrum objects
    spectrum_open, spectrum_closed = get_spectrum(subject_id, input_dir)

    # Create string for the title of the plot
    str_picks = str(picks)

    # Create string for the name of the file
    str_picks_name = str_picks.replace('[', '')
    str_picks_name = str_picks_name.replace(']', '')
    str_picks_name = str_picks_name.replace(' ', '')
    str_picks_name = str_picks_name.replace("'", '')
    str_picks_name = str_picks_name.replace(',', '-')

    # Plot and save the PSDs
    psd_open = spectrum_open.plot(average=True, picks=picks, show=False)
    psd_open.get_axes()[0].set_title('PSD resting state open ' + str_picks)
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATEOPEN', 'psd-plots')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATEOPEN', 'psd-plots'))
    psd_open.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATEOPEN', 'psd-plots', f'sub-{subject_id}_ch-{str_picks_name}_psd-open.png'))

    psd_closed = spectrum_closed.plot(average=True, picks=picks, show=False)
    psd_closed.get_axes()[0].set_title('PSD resting state closed ' + str_picks)
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATECLOSE', 'psd-plots')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATECLOSE', 'psd-plots'))
    psd_closed.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'RESTINGSTATECLOSE', 'psd-plots', f'sub-{subject_id}_ch-{str_picks_name}_psd-closed.png'))

    return psd_open, psd_closed

