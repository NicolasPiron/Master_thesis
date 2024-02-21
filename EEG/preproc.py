import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import glob
from mne.preprocessing import ICA
import pandas as pd
import autoreject
from N2pc.n2pc_func.set_paths import get_paths

def load_data(subject_id, task, input_path, plot_data=True):
    ''' Loads the data from the BIDS folder and concatenates the runs if there is more than 1 run.

    Parameters
    ----------
    subject_id : str
        The subject ID. 2 digits.
    task : str
        The task name. Can be 'N2pc', 'Alpheye', 'RESTINGSTATEOPEN' or 'RESTINGSTATECLOSED'.
    input_path : str
        The path to the BIDS folder (sourcedata).
    plot_data : bool
        Whether to plot the raw data or not. Default is True.
    
    Returns
    -------
    raw : mne.io.Raw
        The raw data.
    e_list : array : only for N2pc and Alpheye (no events for resting state)
        The event list.
    '''

    subject_id = str(subject_id)

    if task == 'RESTINGSTATEOPEN' or task == 'RESTINGSTATECLOSE':

        file = os.path.join(input_path, fr"sub-{subject_id}/ses-01/eeg/sub-{subject_id}_ses-01_task-{task}_eeg.bdf")

        raw = mne.io.read_raw_bdf(file)
        raw.load_data()

        # crop the data to 3 minutes
        raw.crop(tmin=0,tmax=180)

        # Plot the raw data to select bad channels
        if plot_data == True:
            raw.plot(scalings='auto')

            # Pause the script until you press a key
            plt.show(block=False)
            input("Press Enter when all the bad channels have been defined...")
            print('====================== Bad channels defined')

        return raw

    elif task == 'N2pc' or task == 'Alpheye':

        path = fr"sub-{subject_id}/ses-01/eeg/sub-{subject_id}_ses-01_task-{task}_run-"
        files = glob.glob(os.path.join(input_path,path+'*.bdf'))
        files.sort()
        
        # Other option / should work with both tasks
        raws = []
        for file in files:
            raw = mne.io.read_raw_bdf(file)
            raw.load_data()
            raws.append(raw)
        raw = mne.concatenate_raws(raws)

        e_list = mne.find_events(raw, stim_channel='Status')
        annot = mne.annotations_from_events(e_list, raw.info['sfreq'])
        raw = raw.set_annotations(annot)

        # Plot the raw data to select bad channels
        if plot_data == True:
            raw.plot(scalings='auto')

            # Pause the script until you press a key
            plt.show(block=False)
            input("Press Enter when all the bad channels have been defined...")
            print('====================== Bad channels defined')

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
    raw.load_data()

    subject_id = str(subject_id)
    # Drop channels if they are present
    channels_to_drop = ['EXG7', 'EXG8']
    channels_present = [channel for channel in channels_to_drop if channel in raw.info['ch_names']]
    if channels_present:
        raw.drop_channels(channels_present)
        print(f"Dropped channels: {', '.join(channels_present)}")
    else:
        print("No channels to drop.")

    psd_before_filter = raw.plot_psd(fmin=0, fmax=60, dB=True, average=True, show=False)
    # Set channel types, montage and filter
    raw.set_channel_types({'EXG1': 'eog', 'EXG2': 'eog','EXG3': 'eog','EXG4': 'eog', 'EXG5': 'eog','EXG6': 'eog'})
    raw.set_montage('biosemi64') 
    raw.notch_filter([50, 100, 150])
    raw.filter(1.,30., phase='zero-double')
    psd_fig = raw.plot_psd(fmin=0, fmax=60, dB=True, average=True, show=False)
    #plt.close() seems to slow everyting on the server ?_?

    # Save the PSD plot
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-01-psd')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-01-psd'))
        print('Directory created')
    psd_before_filter.savefig(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-01-psd', f'sub-{subject_id}-psd-before-filter-{task}.png'))
    psd_fig.savefig(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-01-psd', f'sub-{subject_id}-psd-{task}.png'))
    print('====================== PSD plot saved')

    # save the list of bad channels
    bad_channels = raw.info['bads']
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-01-bad_channels')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-01-bad_channels'))
        print('Directory created')
    with open(os.path.join(output_path, f'sub-{subject_id}', task,'preprocessing', 'step-01-bad_channels', f'sub-{subject_id}-bad_channels-{task}.txt'), 'w') as f:
        for item in bad_channels:
            f.write("%s\n" % item)
    print('====================== bad channels saved')
    
    # Interpolate bad channels
    mne.io.Raw.interpolate_bads(raw, reset_bads=False)
    if plot_data == True:
        raw.plot(scalings='auto')

    # Save the raw data
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-02-raw-interpolated')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-02-raw-interpolated'))
        print('Directory created')
    raw.save(os.path.join(output_path, f'sub-{subject_id}', task,'preprocessing', 'step-02-raw-interpolated', f'sub-{subject_id}-raw-interpolated-{task}.fif'), overwrite=True)

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
        events = correct_df.values

        event_id = {'dis_top/target_l':1,
            'dis_top/target_r':2,
            'no_dis/target_l':3,
            'no_dis/target_r':4,
            'dis_bot/target_l':5,
            'dis_bot/target_r':6,
            'dis_right/target_l':7,
            'dis_left/target_r':8,
            }
        
        tmin = -0.2
        tmax = 0.8

    elif task == 'Alpheye':

        df = pd.DataFrame(e_list, columns=['timepoint', 'duration', 'stim'])
        df_stim = df[(df['stim'] == 2) | (df['stim'] == 4)].reset_index()
        df_stim = df_stim.add_prefix('img_')
        events = df_stim[['img_timepoint', 'img_duration', 'img_stim']].values

        event_id = {'Landscape':2,
                    'Human':4}
        tmin = -0.2
        tmax = 6.0
    
    elif task == 'RESTINGSTATECLOSE' or 'RESTINGSTATEOPEN':
        
        tmin = 0
        tmax = 2.0
        
        # Define the sampling rate and epoch duration in seconds
        sampling_rate = raw.info['sfreq']
        epoch_duration = tmax - tmin
        
        # Create fake events at regular intervals
        fake_event_samples = np.arange(0, len(raw), int(epoch_duration * sampling_rate))

        events = np.array([fake_event_samples, np.zeros_like(fake_event_samples), np.ones_like(fake_event_samples)]).T
        event_id = {'fake_event':1}
        
        # even if not very useful, transform the fake event list into a df to save it
        df = pd.DataFrame({'timepoint':events[:,0], 'duration':events[:,1], 'stim':events[:,2]})
    
    # Epoch the data
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, event_repeated='drop', preload=True)

    # Set the EEG reference and resample the data
    epochs.set_eeg_reference(ref_channels='average')
    epochs.resample(512)

    # Save the event list and the epochs
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-03-event_lists')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-03-event_lists'))
        print('Directory created')
    df.to_csv(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-03-event_lists', f'sub-{subject_id}-elist-{task}.csv'), index=False)

    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-04-epochs')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-04-epochs'))
        print('Directory created')
    epochs.save(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-04-epochs', f'sub-{subject_id}-epochs-{task}.fif'), overwrite=True)

    return epochs

def automated_epochs_rejection(subject_id, task, epochs, output_path):
    ''' Performs ICA and automated epochs rejection and saves the epochs. Save plots of the rejected epochs and the cleaned average signal.
    There is a first use of Autoreject to fit the ICA to less noisy signal, and a second after the signal has been cleaned with ICA.
    
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
    epochs_clean : mne.Epochs
        The epoched data after cleaning.
    '''
    subject_id = str(subject_id)
    # Perform automated epochs rejection
    ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                           n_jobs=2, verbose=True)
    #ar.fit(epochs[:100]) 
    #ar_epochs, reject_log = ar.transform(epochs, return_log=True)
    ar_epochs, reject_log = ar.fit_transform(epochs, return_log=True)

    # Save the rejected epochs, the log and plot
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-05-ar_epochs-before-ica')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-05-ar_epochs-before-ica'))
        print('Directory created')
    ar_epochs.save(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-05-ar_epochs-before-ica', f'sub-{subject_id}-ar_epochs-before-ica-{task}.fif'), overwrite=True)

    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-05-reject_log-before-ica')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-05-reject_log-before-ica'))
        print('Directory created')
    reject_log.save(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-05-reject_log-before-ica', f'sub-{subject_id}-reject_log-before-ica-{task}.npz'), overwrite=True)

    log_plot = reject_log.plot('horizontal', show=False)
    plt.close(log_plot)
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-05-dropped_epochs-for-ica')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-05-dropped_epochs-for-ica'))
        print('Directory created')
    log_plot.savefig(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-05-dropped_epochs-for-ica', f'sub-{subject_id}-dropped_epochs-for-ica-{task}.png'))

    # Define a function that allows the user to chose the component to exclude
    def get_user_inputs():
        user_inputs = []
        while True:
            user_input = input("Chose a component to exclude (or 'ok' to finish): ")
            if user_input.lower() == 'ok':
                break
            try:
                value = int(user_input)
                user_inputs.append(value)
            except ValueError:
                print("Invalid input. Please enter a valid number or 'ok'.")
        return user_inputs
    
    ica = ICA(n_components=24, random_state=98)
    print('====================== ICA object created')
    ica.fit(epochs[~reject_log.bad_epochs], decim=10)

    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-06-ica')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-06-ica'))
        print('Directory created')
    ica.save(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-06-ica', f'sub-{subject_id}-ica-{task}.fif.gz'), overwrite=True)

    print('====================== fit done')
    ica.plot_components(picks=np.arange(0,10,1))
    print('====================== plot done')
    # Get the user inputs
    user_inputs = get_user_inputs()
    
    # Print which components were chosen
    if user_inputs:
            print('Components excluded :', user_inputs)
    else:
            print("No user inputs.")
      
    if user_inputs == []:
        print('No components excluded')
        epochs_clean = epochs
    else:
        # Exclude components and apply ICA to the epochs (no rejection yet)
        ica.exclude = user_inputs

        # Plot the difference with and without the IC removal
        IC_removal = ica.plot_overlay(epochs.average(), exclude=ica.exclude, show=False)

        if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-06-ic_removal')) == False:
            os.makedirs(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-06-ic_removal'))
            print('Directory created')
        IC_removal.savefig(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-06-ic_removal', f'sub-{subject_id}-ic_removal-{task}.png'))

        # Plot the ICs properties
        ICs_properties = ica.plot_properties(epochs, picks=user_inputs, show=False)

        if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-06-ics_properties')) == False:
            os.makedirs(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-06-ics_properties'))
            print('Directory created')
        for i, fig in enumerate(ICs_properties):
            fig.savefig(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-06-ics_properties', f'sub-{subject_id}-IC0{user_inputs[i]}_properties-{task}.png'))

        
        # Apply ICA to the epochs
        epochs_clean = ica.apply(epochs, exclude=ica.exclude)

    # Fit transform the cleaned epochs to remove the bad epochs that are still bad after ICA
    ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                            n_jobs=2, verbose=True)
    #ar.fit(epochs[:100]) 
    #ar_epochs, final_reject_log = ar.transform(epochs, return_log=True)
    epochs_clean, final_reject_log = ar.fit_transform(epochs_clean, return_log=True)


    if task == 'N2pc':
        # Apply baseline correction AFTER ICA
        epochs_clean = epochs_clean.apply_baseline(baseline=(-0.2,0), verbose=True)

    # Plot the average signal before and after rejecting the epochs and save the plot
    if len(epochs[final_reject_log.bad_epochs]) > 0:
        evoked_bad = epochs[final_reject_log.bad_epochs].average()
        plt.plot(evoked_bad.times, evoked_bad.data.T * 1e6, 'r', zorder=-1)
        clean_plot = epochs_clean.average().plot(axes=plt.gca())
        if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-06-after-ar')) == False:
            os.makedirs(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-06-after-ar'))
            print('Directory created')
        clean_plot.savefig(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-06-after-ar', f'sub-{subject_id}-after-ar-{task}.png'))
    
    # The final epochs we will use for further analysis
    if os.path.exists(os.path.join(output_path , f'sub-{subject_id}', task, 'cleaned_epochs')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', task, 'cleaned_epochs'))
        print('Directory created')
    epochs_clean.save(os.path.join(output_path, f'sub-{subject_id}', task, 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-{task}.fif'), overwrite=True)
    # plot the final reject log - it shows what epochs were rejected after the ICA was applied
    final_log_plot = final_reject_log.plot('horizontal', show=False)
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-06-final-dropped_epochs')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-06-final-dropped_epochs'))
        print('Directory created')
    final_log_plot.savefig(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-06-final-dropped_epochs', f'sub-{subject_id}-final-dropped_epochs-{task}.png'))
    # save the final reject log for further analysis (alpha power df)
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-06-final-reject_log')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-06-final-reject_log'))
        print('Directory created')
    final_reject_log.save(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'step-06-final-reject_log', f'sub-{subject_id}-final-reject_log-{task}.npz'), overwrite=True)
    
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
    first_step_epochs = epochs.average().plot(ylim=ylim, spatial_colors=True, show=False)
    last_step_epochs = epochs_clean.average().plot(ylim=ylim, spatial_colors=True, show=False)

    # Save the plots
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-03-epochs')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-03-epochs'))
        print('Directory created')
    first_step_epochs.savefig(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-03-epochs', f'sub-{subject_id}-epochs-{task}.png'))
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-06-cleaned_epochs')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-06-cleaned_epochs'))
        print('Directory created')
    last_step_epochs.savefig(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-06-cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-{task}.png'))

    if task == 'N2pc':

        # Target on the right, look at the alpha power
        target_r_epochs = epochs_clean["target_r"]
        target_r_evoked = target_r_epochs.average()
        evoked_topo = target_r_evoked.plot_topomap(times=[0.0, 0.1, 0.2, 0.25, 0.3, 0.4], ch_type="eeg", show=False)
    
    elif task == 'Alpheye':

        # Topo for human condition
        human_epochs = epochs_clean["Human"]
        human_evoked = human_epochs.average()
        evoked_topo = human_evoked.plot_topomap(times=[0.0, 0.2, 0.5, 1, 2, 3], ch_type="eeg", show=False)

    elif task == 'RESTINGSTATECLOSE' or 'RESTINGSTATEOPEN':

        # Topo for resting state
        resting_epochs = epochs_clean["fake_event"]
        resting_evoked = resting_epochs.average()
        evoked_topo = resting_evoked.plot_topomap(times=[0.0, 0.5, 1.0, 1.5, 1.9], ch_type="eeg", show=False)
    
    # Save the plots
    if os.path.exists(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-06-evoked_topo')) == False:
        os.makedirs(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-06-evoked_topo'))
        print('Directory created')
    evoked_topo.savefig(os.path.join(output_path, f'sub-{subject_id}', task, 'preprocessing', 'plots', 'step-06-evoked_topo', f'sub-{subject_id}-evoked_topo-{task}.png'))


############################################################################################################
# Additional functions for the secondary resting state pipeline
    

def get_reject_log(raw, duration=1.):
    '''Epochs the raw data and runs the autoreject algorithm.
    Get the reject log from the autoreject algorithm.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data
    duration : float
        Duration of the epochs in seconds

    Returns
    -------
    events : np.array
        Events
    reject_log : autoreject.RejectLog
        Reject log from the autoreject algorithm
    '''

    duration = 1.
    events = mne.make_fixed_length_events(raw,duration=duration)
    epochs = mne.Epochs(raw, events=events, tmin=0, tmax=duration, baseline=None, preload=True)
    ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                            n_jobs=2, verbose=True)
    _, reject_log = ar.fit_transform(epochs, return_log=True)

    return events, reject_log

def get_annotations(reject_log, sfreq, duration, events):
    '''Get annotations from the reject log.
    The annotations are used to mark the epochs of the log as bad segments in the raw data.

    Parameters
    ----------
    reject_log : autoreject.RejectLog
        Reject log from the autoreject algorithm
    sfreq : float
        Sampling frequency
    duration : float
        Duration of the epochs in seconds
    events : np.array
        Events

    Returns
    -------
    annot : mne.Annotations
        Annotations
    '''
     
    log = reject_log.bad_epochs
    bad_epochs = np.where(log==True)
    bad_onsets = []
    for bad in bad_epochs[0]:
        onset = events[bad,0]
        onset = onset/sfreq
        bad_onsets.append(onset)
        
    annot = mne.Annotations(
        onset=bad_onsets,
        duration=[duration]*len(bad_onsets), 
        description=['bad']*len(bad_onsets),
    )

    return annot

def annotate_raw(raw, duration=1.):
    '''Annotate the raw data using the autoreject algorithm.
    The annotations are used to mark the epochs of the log as bad segments in the raw data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data
    duration : float
        Duration of the epochs in seconds

    Returns
    -------
    raw_annoted : mne.io.Raw
        Annotated raw data
    reject_log : autoreject.RejectLog
        Reject log from the autoreject algorithm, important to get the plot later
    '''

    sfreq = raw.info['sfreq']
    events, reject_log = get_reject_log(raw, duration)
    annot = get_annotations(reject_log, sfreq, duration, events)
    raw_annoted = raw.copy().set_annotations(annot)
    
    return raw_annoted, reject_log

def get_user_inputs():
    '''Get user inputs to exclude components from the ICA.
    Used in the run_ica_on_raw function.
    '''
    user_inputs = []
    while True:
        user_input = input("Chose a component to exclude (or 'ok' to finish): ")
        if user_input.lower() == 'ok':
            break
        try:
            value = int(user_input)
            user_inputs.append(value)
        except ValueError:
            print("Invalid input. Please enter a valid number or 'ok'.")
    return user_inputs

def run_ica_on_raw(raw_annoted, raw):
    '''Run ICA on annotated raw data and apply the ICA to the raw data.

    Parameters
    ----------
    raw_annoted : mne.io.Raw
        Annotated raw data  
    raw : mne.io.Raw
        Raw data

    Returns
    -------
    raw_clean : mne.io.Raw
        Clean raw data  
    ICs_properties : matplotlib.figure.Figure
        Plot of the ICs properties  
    '''

    ica = ICA(n_components=24, random_state=98)
    ica.fit(raw_annoted, decim=10)
    ica.plot_components(picks=np.arange(0,10,1))
    user_inputs = get_user_inputs()
    if user_inputs:
            print('Components excluded :', user_inputs)
    else:
            print("No user inputs.")
    if user_inputs == []:
        print('No components excluded')
        raw_clean = raw_annoted
        ICs_properties = []
    else:
        ica.exclude = user_inputs
        raw_clean = ica.apply(raw, exclude=ica.exclude)
        ICs_properties = ica.plot_properties(raw, picks=user_inputs, show=False)
        
    return raw_clean, ICs_properties

def ar_ica_ar(raw):
    '''Run the autoreject algorithm on the raw data, then run ICA and finally run autoreject again.
    This function is used to clean the raw data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data

    Returns
    -------
    raw_clean_annoted : mne.io.Raw
        Clean raw data with annotations
    raw_annoted : mne.io.Raw
        Raw data with annotations before ICA
    IC_properties : matplotlib.figure.Figure
        Plot of the ICs properties
    reject_log1 : autoreject.RejectLog
        Reject log from the autoreject algorithm before ICA
    reject_log2 : autoreject.RejectLog
        Reject log from the autoreject algorithm after ICA
    '''
    raw_annoted, reject_log1 = annotate_raw(raw, duration=1.)
    raw_clean, IC_properties = run_ica_on_raw(raw_annoted, raw)
    raw_clean_annoted , reject_log2 = annotate_raw(raw_clean, duration=1.)

    return raw_clean_annoted, raw_annoted, IC_properties, reject_log1, reject_log2

def clean_raw(subject_id, condition):
    '''Clean the raw data using the autoreject algorithm and ICA.
    Before running this function, you need to have an interpolated and filtered raw file.
    Saves the raw clean, the ICs properties and the reject logs.

    Parameters
    ----------
    subject_id : str or int
        Subject ID : 2 digits (e.g. '01' or '21')
    condition : str
        Condition (e.g. 'RESTINGSTATEOPEN' or 'RESTINGSTATECLOSE')
    
    Returns
    -------
    None
    '''
    input_dir, output_dir = get_paths()
    raw = mne.io.read_raw(os.path.join(input_dir, f'sub-{subject_id}', condition, 'preprocessing',
                                        'step-02-raw-interpolated', f'sub-{subject_id}-raw-interpolated-{condition}.fif'))
    raw.resample(512)
    raw.set_eeg_reference(ref_channels='average')

    raw_clean_annoted, raw_annoted, IC_properties, reject_log1, reject_log2 = ar_ica_ar(raw)

    reject_plot1 = reject_log1.plot(show=False)
    reject_plot2 = reject_log2.plot(show=False)

    for path in ['03-reject-log-before-ica', '04-raw-annot','05-final-reject-log', '06-raw-annot-clean',
                'plots/03-reject-log-before-ica', 'plots/04-ICs', 'plots/05-final-reject-log']:
        if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', condition, 'preprocessing', 'additional', path)):
            os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', condition, 'preprocessing', 'additional', path))
    
    reject_log1.save(os.path.join(output_dir, f'sub-{subject_id}', condition, 'preprocessing', 'additional', '03-reject-log-before-ica', f'sub-{subject_id}-reject-log-before-ica-{condition}.npz'), overwrite=True)
    reject_log2.save(os.path.join(output_dir, f'sub-{subject_id}', condition, 'preprocessing', 'additional', '05-final-reject-log', f'sub-{subject_id}-final-reject-log-{condition}.npz'), overwrite=True)
    reject_plot1.savefig(os.path.join(output_dir, f'sub-{subject_id}', condition, 'preprocessing', 'additional', 'plots/03-reject-log-before-ica', f'sub-{subject_id}-reject-log-before-ica-{condition}.png'))
    reject_plot2.savefig(os.path.join(output_dir, f'sub-{subject_id}', condition, 'preprocessing', 'additional', 'plots/05-final-reject-log', f'sub-{subject_id}-final-reject-log-{condition}.png'))
    raw_clean_annoted.save(os.path.join(output_dir, f'sub-{subject_id}', condition, 'preprocessing', 'additional', '06-raw-annot-clean', f'sub-{subject_id}-raw-annot-clean-{condition}.fif'), overwrite=True)
    raw_annoted.save(os.path.join(output_dir, f'sub-{subject_id}', condition, 'preprocessing', 'additional', '04-raw-annot', f'sub-{subject_id}-raw-annot-{condition}.fif'), overwrite=True)
    if IC_properties:
        for i, fig in enumerate(IC_properties):
            fig.savefig(os.path.join(output_dir, f'sub-{subject_id}', condition, 'preprocessing', 'additional', 'plots/04-ICs', f'sub-{subject_id}-IC{i}-{condition}.png'))


def ten_sec_epochs(subject_id, condition):

    input_dir, output_dir = get_paths()
    raw = mne.io.read_raw(os.path.join(input_dir,  f'sub-{subject_id}', condition, 'preprocessing', 'additional', '06-raw-annot-clean', f'sub-{subject_id}-raw-annot-clean-{condition}.fif'))
    duration = 10.
    events = mne.make_fixed_length_events(raw,duration=duration)
    epochs = mne.Epochs(raw, events=events, tmin=0, tmax=10., baseline=None, preload=True, reject_by_annotation=True)

    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', condition, 'preprocessing', 'additional', '07-epochs-10s')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', condition, 'preprocessing', 'additional', '07-epochs-10s'))
    epochs.save(os.path.join(output_dir, f'sub-{subject_id}', condition, 'preprocessing', 'additional', '07-epochs-10s', f'sub-{subject_id}-{condition}-10s-epo.fif'), overwrite=True)
