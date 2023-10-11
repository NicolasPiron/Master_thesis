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
                    if freq is alpha_freqs:
                        freq_ = 'alpha'
                    elif freq is theta_freqs:
                        freq_ = 'theta'
                    
                    if condition is 'RESTINGSTATEOPEN':
                        condition_ = 'open'
                    elif condition is 'RESTINGSTATECLOSE':
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
