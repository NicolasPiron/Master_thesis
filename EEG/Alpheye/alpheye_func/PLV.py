import seaborn as sns
from mne_connectivity import spectral_connectivity_time
import numpy as np
import mne
import os
import matplotlib.pyplot as plt

def PLV_sliding_window(subjects='all', condition='all', freqs=(8,12), indices=[6, 4]):
    ''' Plots the PLV over time 
    
    Parameters
    ----------
    subjects: str
    
    condition: str
    
    freqs: tuple
    
    indices: list
    
    Returns
    ----------
    
    
    '''
    # load data
    if subjects == 'all':
        
        print('Reading data from all subjects')
        epochs = mne.read_epochs('/Users/nicolaspiron/Documents/PULSATION/Data/EEG/Alpheye/data-preproc-alpheye/epochs_all/S_all_subj_alpheye_.fif')
        
    else:
        
        print(f'Reading data from subject {subjects}')
        path_epochs = '/Users/nicolaspiron/Documents/PULSATION/Data/EEG/Alpheye/data-preproc-alpheye/total_epochs'
        epochs = mne.read_epochs(os.path.join(path_epochs, subjects + '_total_epochs.fif'))
    
    # chose condition
    if condition == 'all':    
        
        print('Computing PLV for all conditions together...')
    
    else:
        
        print(f'Computing PLV for condition {condition}...')
        epochs = epochs[condition]
    
    # Crop epochs to get rid of the baseline part (200ms) 
    #start = 0.0
    #stop = epochs.tmax
    #epochs_cropped = epochs.crop(tmin=start, tmax=stop)

    # Define the time window size and step size for the sliding window
    win_size = 1.5  # in seconds
    step_size = 0.1  # in seconds

    # Initialize an empty list to store the connectivity values over time
    con_vals = []

    
    # Convert list of indices into a tuple of separated lists
    idx = ([indices[0]], [indices[1]])
    
    # Loop over each time step in the data
    for tmin in np.arange(0, epochs.times[-1] - win_size, step_size):
        # Extract the time window centered on this time step
        tmax = tmin + win_size
        epoch_win = epochs.copy().crop(tmin=tmin, tmax=tmax)

        # Compute the connectivity between the two electrodes of interest within this window
        con = spectral_connectivity_time(
            epoch_win, method='plv', sfreq=epochs.info['sfreq'], indices=idx,
                    freqs=freqs, faverage=True, n_jobs=1).get_data()

        # Take the mean connectivity value as the connectivity measure for this time step
        con_val = np.mean(con)

        # Append the connectivity value to the time series
        con_vals.append(con_val)

    plt.plot(con_vals, marker='o', linestyle='-', color='orange')
    plt.ylim(0, 1)
    plt.title(f'{subjects}, freqs (Hz): {freqs}, channels: {[epochs.ch_names[i] for i in indices]}, condition: {condition}')
    plt.xlabel('time steps')
    plt.ylabel('PLV')
    plt.grid(True)

    
    # save the plot
    plt_path = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/PLV_plots/'
    fig_name = plt_path + subjects +'_'+ condition +'_' + str(freqs[0]) + '-' + str(freqs[1]) + 'Hz_' + str(epochs.ch_names[indices[0]]) + '_' + str(epochs.ch_names[indices[1]]) + '.png'
    plt.savefig(fig_name)

    plt.show()
    
    return con_vals