import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import glob
from mne.preprocessing import ICA
import pandas as pd
import autoreject

def load_data(subject_id, task, input_path, plot_data=True):
    ''' Loads the data from the BIDS folder and concatenates the two runs.

    Parameters
    ----------
    subject_id : str
        The subject ID.
    task : str
        The task name. Default is 'N2pc'.
    input_path : strs
        The path to the BIDS folder. Default is '/Users/nicolaspiron/Documents/Master_thesis/EEG/toBIDS/BIDS_data/sourcedata/'.
    plot_data : bool
        Whether to plot the raw data or not. Default is True.
    
    Returns
    -------
    raw : mne.io.Raw
        The raw data.
    e_list : array
        The event list.
    '''
    subject_id = str(subject_id)
    path = fr"sub-{subject_id}/ses-01/eeg/sub-{subject_id}_ses-01_task-{task}_run-"

    # Load the data and concatenate the two runs
    files = glob.glob(os.path.join(input_path,path+'*.bdf'))
    files.sort()
    
    # Other option / should work with both tasks
    raws = []
    for file in files:
        raw = mne.io.read_raw_bdf(file)
        raw.load_data()
        raws.append(raw)
    raw = mne.concatenate_raws(raws)

    # Find the events and create annotations
    e_list = mne.find_events(raw, stim_channel='Status')
    annot = mne.annotations_from_events(e_list, raw.info['sfreq'])
    raw = raw.set_annotations(annot)

    # Plot the raw data to select bad channels
    if plot_data == True:
        raw.plot(scalings='auto')

        # Pause the script until you press a key
        plt.show(block=False)
        input("Press Enter when all the bad channels have been defined...")
        print('bad channels defined')

    return raw, e_list

def filter_and_interpolate(subject_id, task, raw, output_path, plot_data=True):
    ''' Drops useless channels, sets the channel types and the montage, filters the data and interpolates bad channels. 
        Saves the raw data and the PSD plot. Also saves a list with the bad channels that were interpolated.
    
    Parameters
    ----------
    subject_id : str
        The subject ID.
    task : str
        The task name.
    raw : mne.io.Raw 
       The raw data.
    output_path : str
        The path to the output folder.
    plot_data : bool
        Whether to plot the raw data or not. Default is True.
    
    Returns
    -------
    raw : mne.io.Raw
        The filtered and interpolated raw data.            
        '''
    subject_id = str(subject_id)
    # Drop channels if they are present
    channels_to_drop = ['EXG7', 'EXG8']
    channels_present = [channel for channel in channels_to_drop if channel in raw.info['ch_names']]
    if channels_present:
        raw.drop_channels(channels_present)
        print(f"Dropped channels: {', '.join(channels_present)}")
    else:
        print("No channels to drop.")

    # Set channel types, montage and filter
    raw.set_channel_types({'EXG1': 'eog', 'EXG2': 'eog','EXG3': 'eog','EXG4': 'eog', 'EXG5': 'eog','EXG6': 'eog'})
    raw.set_montage('biosemi64') 
    raw.notch_filter([50, 100, 150])
    raw.filter(1.,30., phase='zero-double')
    psd_fig = raw.plot_psd(fmin=0, fmax=50, dB=True, average=True)

    # Save the PSD plot
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-01-psd')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}','preprocessing', 'plots', 'step-01-psd'))
        print('Directory created')
    psd_fig.savefig(os.path.join(output_path, f'sub-{subject_id}','preprocessing', 'plots', 'step-01-psd', f'sub-{subject_id}-psd-{task}.png'))
    
    # save the list of bad channels
    bad_channels = raw.info['bads']
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}','preprocessing', 'step-01-bad_channels')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}','preprocessing', 'step-01-bad_channels'))
        print('Directory created')
    with open(os.path.join(output_path, f'sub-{subject_id}','preprocessing', 'step-01-bad_channels', f'sub-{subject_id}-bad_channels-{task}.txt'), 'w') as f:
        for item in bad_channels:
            f.write("%s\n" % item)
    
    # Interpolate bad channels
    mne.io.Raw.interpolate_bads(raw, reset_bads=False)
    if plot_data == True:
        raw.plot(scalings='auto')

    # Save the raw data
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}','preprocessing', 'step-02-raw-interpolated')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}','preprocessing', 'step-02-raw-interpolated'))
        print('Directory created')
    raw.save(os.path.join(output_path, f'sub-{subject_id}','preprocessing', 'step-02-raw-interpolated', f'sub-{subject_id}-raw-interpolated-{task}.fif'), overwrite=True)

    return raw

def epoch_data(subject_id, task, raw, e_list,  output_path):
    ''' Epochs the data and saves the epochs and the event list.
    
    Parameters
    ----------
    subject_id : str
        The subject ID.
    task : str
        The task name.
    raw : mne.io.Raw
        The raw data.
    e_list : array
        The event list.
    output_path : str
        The path to the output folder.
    
    Returns
    -------
    epochs : mne.Epochs
        The epoched data.
    '''
    subject_id = str(subject_id)
    if task == 'N2pc':

        # Convert the event list to a dataframe and select the events of interest
        df = pd.DataFrame(e_list, columns=['timepoint', 'duration', 'stim'])
        mask = (df['stim'].isin(range(1, 9))) & (df['stim'].shift(-1) == 128) | (df['stim'] == 128) & (df['stim'].shift(1).isin(range(1,9)))
        correct_df = df[mask]
        mne_events = correct_df.values

        event_dict = {'dis_top/target_l':1,
              'dis_top/target_r':2,
              'no_dis/target_l':3,
              'no_dis/target_r':4,
              'dis_bot/target_l':5,
              'dis_bot/target_r':6,
              'dis_right/target_l':7,
              'dis_left/target_r':8,
             }
        
        # Epoch the data
        epochs = mne.Epochs(raw, mne_events, event_id=event_dict, tmin=-0.2, tmax=0.8, baseline=(-0.2,0), event_repeated='drop', preload=True)
        
    elif task == 'Alpheye':
        
        df = pd.DataFrame(e_list, columns=['timepoint', 'duration', 'stim'])
        df_stim = df[(df['stim'] == 2) | (df['stim'] == 4)].reset_index()
        df_stim = df_stim.add_prefix('img_')
        mne_events = df_stim[['img_timepoint', 'img_duration', 'img_stim']].values
        
        event_dict = {'Landscape':2,
                    'Human':4}

        epochs = mne.Epochs(raw, mne_events, event_id=event_dict, tmin=-0.2, tmax=6, baseline=(-0.2, 0), event_repeated='drop', preload=True)

    # Set the EEG reference and resample the data
    epochs.set_eeg_reference(ref_channels='average')
    epochs.resample(512)
    
    # Save the event list and the epochs
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'step-03-event_lists')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'step-03-event_lists'))
        print('Directory created')
    df.to_csv(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'step-03-event_lists', f'sub-{subject_id}-elist-{task}.csv'), index=False)

    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'step-04-epochs')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'step-04-epochs'))
        print('Directory created')
    epochs.save(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'step-04-epochs', f'sub-{subject_id}-epochs-{task}.fif'), overwrite=True)

    return epochs

def automated_epochs_rejection(subject_id, task, epochs, output_path):
    ''' Performs automated epochs rejection and saves the epochs. Save plots of the rejected epochs and the cleaned average signal.
    
    Parameters
    ----------
    subject_id : str
        The subject ID.
    task : str
        The task name.
    epochs : mne.Epochs
        The epoched data.
    output_path : str
        The path to the output folder.
    
    Returns
    -------
    epochs : mne.Epochs
        The epoched data with rejected epochs.
    reject_log : autoreject.RejectLog
        The log of the rejected epochs.
    '''
    subject_id = str(subject_id)
    # Perform automated epochs rejection
    ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                           n_jobs=2, verbose=True)
    ar_epochs, reject_log = ar.fit_transform(epochs, return_log=True)

    # Save the rejected epochs, the log and plot
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'step-05-ar_epochs-before-ica')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'step-05-ar_epochs-before-ica'))
        print('Directory created')
    ar_epochs.save(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'step-05-ar_epochs-before-ica', f'sub-{subject_id}-ar_epochs-before-ica-{task}.fif'), overwrite=True)

    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'step-05-reject_log-before-ica')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'step-05-reject_log-before-ica'))
        print('Directory created')
    reject_log.save(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'step-05-reject_log-before-ica', f'sub-{subject_id}-reject_log-before-ica-{task}.npz'), overwrite=True)

    log_plot = reject_log.plot('horizontal')
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-05-dropped_epochs-for-ica')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-05-dropped_epochs-for-ica'))
        print('Directory created')
    log_plot.savefig(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-05-dropped_epochs-for-ica', f'sub-{subject_id}-dropped_epochs-for-ica-{task}.png'))


    #return ar_epochs, reject_log

#def clean_by_ICA(subject_id, task, epochs, reject_log, output_path):
 #   ''' Applies ICA to the data and saves the cleaned epochs and the ICA plots.
  #  IMPORTANT : we run the Autoreject again, after the ICA was applied. 
   # The first Autoreject is run on the data before ICA, only to fit the ICA. 
    
    #Parameters
    #----------
    #subject_id : str
    #    The subject ID.
    #task : str
    #    The task name.
    #ar_epochs : mne.Epochs
    #    The epoched data with rejected epochs.
    #epochs : mne.Epochs
    #    The epoched data without rejection

    #Returns
    #-------
    #epochs_clean : mne.Epochs
     #   The epoched data after cleaning.
    #'''
    #subject_id = str(subject_id)
    # Define a function that allows the user to chose the component to exclude
    def get_user_inputs():
        user_inputs = []
        while True:
            user_input = input("Chose a component to exclude (or 'ok' to finish): ")
            if user_input.lower() == 'ok':
                break
            try:
                value = float(user_input)
                user_inputs.append(value)
            except ValueError:
                print("Invalid input. Please enter a valid number or 'ok'.")
        return user_inputs
    
    ica = ICA(random_state=99)
    ica.fit(epochs[~reject_log.bad_epochs])
    ica.plot_components(picks=np.arange(0,10,1))
    # Get the user inputs
    user_inputs = get_user_inputs()
    
    # Print which components were chosen
    if user_inputs:
            print('Components excluded :', user_inputs)
    else:
            print("No user inputs.")
      
    # Exclude components and apply ICA to the epochs (no rejection yet)
    ica.exclude = user_inputs
    IC_removal = ica.plot_overlay(epochs.average(), exclude=ica.exclude)
    epochs_clean = ica.apply(epochs, exclude=ica.exclude)

    # Fit transform the cleaned epochs to remove the bad epochs that are still bad after ICA
    ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                            n_jobs=2, verbose=True)
    epochs_clean, final_reject_log = ar.fit_transform(epochs_clean, return_log=True)

    # Apply baseline correction again because ICA changes the data
    epochs_clean = epochs_clean.apply_baseline(baseline=(-0.2,0), verbose=True)

    # Plot the average signal before and after rejecting the epochs and save the plot
    evoked_bad = epochs[reject_log.bad_epochs].average()
    plt.plot(evoked_bad.times, evoked_bad.data.T * 1e6, 'r', zorder=-1)
    clean_plot = ar_epochs.average().plot(axes=plt.gca())
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-06-wo_bad_epochs')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-06-wo_bad_epochs'))
        print('Directory created')
    clean_plot.savefig(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-06-wo_bad_epochs', f'sub-{subject_id}-wo_bad_epochs-{task}.png'))
    
    # For some reason, the ICA cannot be saved
    # The ICA
    
    if os.path.exists(os.path.join(output_path, 'preprocessing', 'step-06-ica')) == False:
        os.makedirs(os.path.join(output_path, 'preprocessing', 'step-06-ica'))
        print('Directory created')
    # ica.save(os.path.join(output_path, 'preprocessing', 'step-06-ica', f'sub-{subject_id}-ica.fif'), overwrite=True)
    # The final epochs we will use for further analysis
    if os.path.exists(os.path.join(output_path , f'sub-{subject_id}', 'cleaned_epochs')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', 'cleaned_epochs'))
        print('Directory created')
    epochs_clean.save(os.path.join(output_path, f'sub-{subject_id}', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-{task}.fif'), overwrite=True)
    # plot to see how it looks with and without the IC removal
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-06-ic_removal')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-06-ic_removal'))
        print('Directory created')
    IC_removal.savefig(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-06-ic_removal', f'sub-{subject_id}-ic_removal-{task}.png'))
    # plot the final reject log - it shows what epochs were rejected after the ICA was applied
    final_log_plot = final_reject_log.plot('horizontal')
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-06-final-dropped_epochs')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-06-final-dropped_epochs'))
        print('Directory created')
    final_log_plot.savefig(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-06-final-dropped_epochs', f'sub-{subject_id}-final-dropped_epochs-{task}.png'))
    # save the final reject log for further analysis (alpha power df)
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'step-06-final-reject_log')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'step-06-final-reject_log'))
        print('Directory created')
    final_reject_log.save(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'step-06-final-reject_log', f'sub-{subject_id}-final-reject_log-{task}.npz'), overwrite=True)
    
    return epochs_clean

def quality_check_plots(subject_id, task, epochs, epochs_clean, output_path):
    ''' Plots the data before and after cleaning and the topography of the evoked signal.
    
    Parameters
    ----------
    subject_id : str
        The subject ID.
    task : str
        The task name.
    epochs : mne.Epochs
        The epoched data.
    epochs_clean : mne.Epochs
        The epoched data after cleaning.
    output_path : str 
        The path to the output folder.

    Returns
    -------
    None
    '''
    subject_id = str(subject_id)
    # Plot the data before and after cleaning
    ylim = dict(eeg=(-15, 15))
    first_step_epochs = epochs.average().plot(ylim=ylim, spatial_colors=True)
    last_step_epochs = epochs_clean.average().plot(ylim=ylim, spatial_colors=True)

    # Save the plots
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-03-epochs')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-03-epochs'))
        print('Directory created')
    first_step_epochs.savefig(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-03-epochs', f'sub-{subject_id}-epochs-{task}.png'))
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-06-cleaned_epochs')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-06-cleaned_epochs'))
        print('Directory created')
    last_step_epochs.savefig(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-06-cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-{task}.png'))

    if task == 'N2pc':

        # Target on the right, look at the alpha power
        target_r_epochs = epochs_clean["target_r"]
        target_r_evoked = target_r_epochs.average()
        evoked_topo = target_r_evoked.plot_topomap(times=[0.0, 0.1, 0.2, 0.25, 0.3, 0.4], ch_type="eeg")
    
    elif task == 'Alpheye':

        # Topo for human condition
        human_epochs = epochs_clean["Human"]
        human_evoked = human_epochs.average()
        evoked_topo = human_evoked.plot_topomap(times=[0.0, 0.2, 0.5, 1, 2, 3], ch_type="eeg")

    # Save the plots
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-06-evoked_topo')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-06-evoked_topo'))
        print('Directory created')
    evoked_topo.savefig(os.path.join(output_path, f'sub-{subject_id}', 'preprocessing', 'plots', 'step-06-evoked_topo', f'sub-{subject_id}-evoked_topo-{task}.png'))


