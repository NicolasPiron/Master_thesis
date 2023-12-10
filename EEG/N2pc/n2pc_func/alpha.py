import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import mne
import os
import re

### ================================================================================================
### ========================================== PSD  ================================================
### ================================================================================================

def get_psd_conditions_single_subj(subject_id, input_dir):
    '''
    Compute the spectrum for each condition for a single subject. Save the spectrum in npy format + the info in fif format.
    Parameters
    ----------
    subject_id : str
        The subject ID to plot. 2 digits format (e.g. 01).
    input_dir : str
        The path to the directory containing the input data.

    Returns
    -------
    psd_dict : dict
        A dictionary containing the PSD for each condition.
        {'dis_mid_target_l': dis_mid_target_l, ...}

    '''
    
    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))
    epochs.info['bads'] = []
    # crop the epochs to the relevant time window
    tmin = -0.2
    tmax = 0.8
    epochs.crop(tmin=tmin, tmax=tmax)

    dis_mid_target_l = epochs['dis_top/target_l','dis_bot/target_l']
    dis_mid_target_r = epochs['dis_top/target_r','dis_bot/target_r']
    no_dis_target_l = epochs['no_dis/target_l']
    no_dis_target_r = epochs['no_dis/target_r']
    dis_right_target_l = epochs['dis_right/target_l']
    dis_left_target_r = epochs['dis_left/target_r']

    # compute the spectrum for each bin
    psd_dict = {}
    for bin_, epochs in zip(['dis_mid_target_l', 'dis_mid_target_r', 'no_dis_target_l', 'no_dis_target_r', 'dis_right_target_l', 'dis_left_target_r', 'all'],
                                [dis_mid_target_l, dis_mid_target_r, no_dis_target_l, no_dis_target_r, dis_right_target_l, dis_left_target_r, epochs]):
        psd_dict[bin_] = epochs.compute_psd(fmin=1, fmax=30).average()
        freqs = psd_dict[bin_].freqs

    
    # add 3 conditions that are the average of the 2 sides of each cond (dis_mid, no_dis, dis_lat)
    # we need to swap the sides when the target is on the right side -> so it's like the target is on the left side

    pairs = [[psd_dict['dis_mid_target_l'], psd_dict['dis_mid_target_r']],
            [psd_dict['no_dis_target_l'], psd_dict['no_dis_target_r']],
            [psd_dict['dis_right_target_l'], psd_dict['dis_left_target_r']]]

    # find right and left channels
    ch_names = psd_dict['dis_mid_target_l'].ch_names
    LCh = []
    RCh = []
    for i, ch in enumerate(ch_names):
        if str(ch[-1]) == 'z':
            print(f'central channel {ch} -> not included in lateral channels list')
        elif int(ch[-1]) % 2 == 0:
            RCh.append(i)
        elif int(ch[-1]) %2 != 2:
            LCh.append(i) 

    pair_names = ['dis_mid', 'no_dis', 'dis_lat']
    for i, pair in enumerate(pairs):
    # the right target psd will be laterally swapped so it is like the target is on the left
        to_swap = pair[1]
        data = to_swap.get_data()
        swapped_data = data.copy()
        swapped_data[RCh] = data[LCh]
        swapped_data[LCh] = data[RCh]
        average = np.mean([pair[0].get_data(), swapped_data], axis=0)
        swapped_psd = mne.time_frequency.SpectrumArray(average, to_swap.info, freqs)
        psd_dict[pair_names[i]] = swapped_psd

    # save each spectrum in npy format (idk why but I can't retrieve the spectrum from hdf5 files)
    # save the info as well
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-data', 'spectrum')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-data', 'spectrum'))
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-data', 'freqs')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-data', 'freqs'))
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-data', 'info')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-data', 'info'))

    for bin_, spectrum in psd_dict.items():
        np.save(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-data', 'spectrum', f'sub-{subject_id}-psd-{bin_}.npy'), spectrum.get_data())
        np.save(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-data', 'freqs', f'sub-{subject_id}-psd-freqs.npy'), spectrum.freqs)
        spectrum.info.save(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-data', 'info', f'sub-{subject_id}-psd-info-{bin_}.fif'))
        print(f'====================== spectral data saved for {subject_id} - {bin_}')

    return psd_dict

def load_psd_single_subj(subject_id, input_dir):
    '''
    Load the PSD data for each condition for a single subject.

    Parameters
    ----------
    subject_id : str
        The subject ID to plot. 2 digits format (e.g. 01).
    input_dir : str
        The path to the directory containing the input data.

    Returns
    -------
    psd_dict : dict
        A dictionary containing the PSD for each condition.
        {'dis_mid_target_l': dis_mid_target_l, ...}

    '''
    if os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-data', 'spectrum', f'sub-{subject_id}-psd-all.npy')):
        psd_dict = {}
        for bin_ in ['dis_mid_target_l', 'dis_mid_target_r', 'no_dis_target_l', 'no_dis_target_r', 'dis_right_target_l', 'dis_left_target_r', 'dis_mid', 'no_dis', 'dis_lat', 'all']:
            spec = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-data', 'spectrum', f'sub-{subject_id}-psd-{bin_}.npy'))
            info = mne.io.read_info(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-data', 'info', f'sub-{subject_id}-psd-info-{bin_}.fif'))
            freqs = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-data', 'freqs', f'sub-{subject_id}-psd-freqs.npy'))
            spec_object = mne.time_frequency.SpectrumArray(spec, info, freqs)
            psd_dict[bin_] = spec_object
            print(f'====================== spectral data loaded for {subject_id} - {bin_}')
    else:
        psd_dict = get_psd_conditions_single_subj(subject_id, input_dir)

    return psd_dict

def get_psd_condition_population(input_dir, output_dir, subject_list, population):
    '''
    Compute the spectrum for each participant, then averages the spectra objects by condition
    and plot the mean topography of the scalp for the population.
    
    Parameters
    ----------
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str 
        The path to the directory where the output will be saved.
    population : str
        Population (thal_control, old_controls, young_controls or pulvinar).

    Returns
    -------
    None
    '''
    print(f'====================== computing spectral data for {population}')
    # create list to store the spectra data (np.arrays)
    dis_mid_target_l_list = []
    dis_mid_target_r_list = []
    no_dis_target_l_list = []
    no_dis_target_r_list = []
    dis_right_target_l_list = []
    dis_left_target_r_list = []
    dis_mid_list = []
    no_dis_list = []
    dis_lat_list = []
    all_list = []

    # loop over the subjects and append the spectra data to the lists
    for subject in subject_list:

        psd_dict = load_psd_single_subj(subject, input_dir)

        # append the spectra data to the lists
        dis_mid_target_l_list.append(psd_dict['dis_mid_target_l'].get_data())
        dis_mid_target_r_list.append(psd_dict['dis_mid_target_r'].get_data())
        no_dis_target_l_list.append(psd_dict['no_dis_target_l'].get_data())
        no_dis_target_r_list.append(psd_dict['no_dis_target_r'].get_data())
        dis_right_target_l_list.append(psd_dict['dis_right_target_l'].get_data())
        dis_left_target_r_list.append(psd_dict['dis_left_target_r'].get_data())
        dis_mid_list.append(psd_dict['dis_mid'].get_data())
        no_dis_list.append(psd_dict['no_dis'].get_data())
        dis_lat_list.append(psd_dict['dis_lat'].get_data())
        all_list.append(psd_dict['all'].get_data())

        # get the frequency axis
        freqs = psd_dict['dis_mid_target_l'].freqs
        print(f'====================== spectral data loaded for {subject}')

    # combine the spectra data
    dis_mid_target_l_combined = np.mean(dis_mid_target_l_list, axis=0)
    dis_mid_target_r_combined = np.mean(dis_mid_target_r_list, axis=0)
    no_dis_target_l_combined = np.mean(no_dis_target_l_list, axis=0)
    no_dis_target_r_combined = np.mean(no_dis_target_r_list, axis=0)
    dis_right_target_l_combined = np.mean(dis_right_target_l_list, axis=0)
    dis_left_target_r_combined = np.mean(dis_left_target_r_list, axis=0)
    dis_mid_combined = np.mean(dis_mid_list, axis=0)
    no_dis_combined = np.mean(no_dis_list, axis=0)
    dis_lat_combined = np.mean(dis_lat_list, axis=0)
    all_combined = np.mean(all_list, axis=0)
    print(f'====================== spectral data combined for {population}')

    spectrum_dict = {'dis_mid_target_l': dis_mid_target_l_combined, 'dis_mid_target_r': dis_mid_target_r_combined, 'no_dis_target_l': no_dis_target_l_combined,
                    'no_dis_target_r': no_dis_target_r_combined, 'dis_right_target_l': dis_right_target_l_combined, 'dis_left_target_r': dis_left_target_r_combined,
                        'dis_mid': dis_mid_combined, 'no_dis': no_dis_combined, 'dis_lat': dis_lat_combined, 'all': all_combined}


    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'psd', 'psd-data', 'spectrum', population)):
            os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'psd', 'psd-data', 'spectrum', population))
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'psd', 'psd-data', 'freqs', population)):
            os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'psd', 'psd-data', 'freqs', population))
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'psd', 'psd-data', 'info', population)):
            os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'psd', 'psd-data', 'info', population))

    for bin_, spectrum in spectrum_dict.items():
        try:
            np.save(os.path.join(output_dir, 'all_subj', 'N2pc', 'psd', 'psd-data', 'spectrum', population, f'{population}-psd-{bin_}.npy'), spectrum)
            np.save(os.path.join(output_dir, 'all_subj', 'N2pc', 'psd', 'psd-data', 'freqs', population, f'{population}-psd-freqs.npy'), freqs)
            psd_dict[bin_].info.save(os.path.join(output_dir, 'all_subj', 'N2pc', 'psd', 'psd-data', 'info', population, f'{population}-psd-info-{bin_}.fif'))
            print(f'====================== spectral data saved for {population} - {bin_}')
        except:
            print(f'====================== spectral data not saved for {population} - {bin_}')
            continue

    # back to spectrum objects if we use this function to get the dict (i.e if we don't pass through the load_psd_population function)
    dis_mid_target_l_combined = mne.time_frequency.SpectrumArray(dis_mid_target_l_combined, info=psd_dict['dis_mid_target_l'].info, freqs=freqs)
    dis_mid_target_r_combined = mne.time_frequency.SpectrumArray(dis_mid_target_r_combined, info=psd_dict['dis_mid_target_r'].info, freqs=freqs)
    no_dis_target_l_combined = mne.time_frequency.SpectrumArray(no_dis_target_l_combined, info=psd_dict['no_dis_target_l'].info, freqs=freqs)
    no_dis_target_r_combined = mne.time_frequency.SpectrumArray(no_dis_target_r_combined, info=psd_dict['no_dis_target_r'].info, freqs=freqs)
    dis_right_target_l_combined = mne.time_frequency.SpectrumArray(dis_right_target_l_combined, info=psd_dict['dis_right_target_l'].info, freqs=freqs)
    dis_left_target_r_combined = mne.time_frequency.SpectrumArray(dis_left_target_r_combined, info=psd_dict['dis_left_target_r'].info, freqs=freqs)
    dis_mid_combined = mne.time_frequency.SpectrumArray(dis_mid_combined, info=psd_dict['dis_mid'].info, freqs=freqs)
    no_dis_combined = mne.time_frequency.SpectrumArray(no_dis_combined, info=psd_dict['no_dis'].info, freqs=freqs)
    dis_lat_combined = mne.time_frequency.SpectrumArray(dis_lat_combined, info=psd_dict['dis_lat'].info, freqs=freqs)
    all_combined = mne.time_frequency.SpectrumArray(all_combined, info=psd_dict['all'].info, freqs=freqs)
    print(f'====================== spectral data converted to SpectrumArray for {population}')

    spectrum_dict = {'dis_mid_target_l': dis_mid_target_l_combined, 'dis_mid_target_r': dis_mid_target_r_combined, 'no_dis_target_l': no_dis_target_l_combined,
                    'no_dis_target_r': no_dis_target_r_combined, 'dis_right_target_l': dis_right_target_l_combined, 'dis_left_target_r': dis_left_target_r_combined,
                        'dis_mid': dis_mid_combined, 'no_dis': no_dis_combined, 'dis_lat': dis_lat_combined, 'all': all_combined}

    return spectrum_dict

def load_psd_population(input_dir, output_dir, subject_list, population):
    '''
    Load the PSD data for each condition for a population.

    Parameters
    ----------
    input_dir : str
        The path to the directory containing the input data.
    population : str
        Population (thal_control, old_controls, young_controls or pulvinar).

    Returns
    -------
    psd_dict : dict
        A dictionary containing the PSD for each condition.
        {'dis_mid_target_l': dis_mid_target_l, ...}

    '''
    if os.path.exists(os.path.join(input_dir, 'all_subj', 'N2pc', 'psd', 'psd-data', 'spectrum', population, f'{population}-psd-all.npy')):
        print(f'====================== spectral data already computed for {population}')
        psd_dict = {}
        for bin_ in ['dis_mid_target_l', 'dis_mid_target_r', 'no_dis_target_l', 'no_dis_target_r', 'dis_right_target_l', 'dis_left_target_r', 'dis_mid', 'no_dis', 'dis_lat', 'all']:
            spec = np.load(os.path.join(input_dir, 'all_subj', 'N2pc', 'psd', 'psd-data', 'spectrum', population, f'{population}-psd-{bin_}.npy'))
            info = mne.io.read_info(os.path.join(input_dir, 'all_subj', 'N2pc', 'psd', 'psd-data', 'info', population, f'{population}-psd-info-{bin_}.fif'))
            freqs = np.load(os.path.join(input_dir, 'all_subj', 'N2pc', 'psd', 'psd-data', 'freqs', population, f'{population}-psd-freqs.npy'))
            spec_object = mne.time_frequency.SpectrumArray(spec, info, freqs)
            psd_dict[bin_] = spec_object    
    else:
        print(f'====================== spectral data not computed for {population}')
        psd_dict = get_psd_condition_population(input_dir, output_dir, subject_list, population)

    print(f'====================== spectral data loaded for {population}')

    return psd_dict

def plot_spectral_topo_single_subj(subject_id, input_dir, output_dir):
    '''
    Parameters
    ----------
    subject_id : str
        The subject ID to plot. 2 digits format (e.g. 01).
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.

    Returns
    -------
    None
    '''

    print(f'====================== plotting spectral topo for {subject_id}')

    spectrum_dict = load_psd_single_subj(subject_id, input_dir)

    # plot the spectrum for theta, alpha, low and high beta frequency bands
    bands = {'Theta (4-8 Hz)': (4, 8), 'Alpha (8-12 Hz)': (8, 12), 'low_Beta (12-30 Hz)': (12, 16), 'high_Beta (16-30 Hz)': (16, 30)}

    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'spectral-topo')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'spectral-topo'))
        print(f'====================== spectral topo dir created for {subject_id}')
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-data')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-data'))
        print(f'====================== psd dir created for {subject_id}')

    for bin_, spectrum in spectrum_dict.items():
        plot = spectrum.plot_topomap(bands=bands, res=512, show=False)
        plot.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'spectral-topo', f'sub-{subject_id}-spectral-topo-{bin_}.png'))
        print(f'====================== spectral topo plotted for {subject_id} - {bin_}')


def plot_spectral_topo_population(input_dir, output_dir, subject_list, population):

    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'spectral-topo', population)):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'spectral-topo', population))

    spectrum_dict = load_psd_population(input_dir, output_dir, subject_list, population)

    
    bands = {'Theta (4-8 Hz)': (4, 8), 'Alpha (8-12 Hz)': (8, 12), 'low_Beta (12-30 Hz)': (12, 16), 'high_Beta (16-30 Hz)': (16, 30)}
    for bin_, spectrum in spectrum_dict.items():
        plot = spectrum.plot_topomap(bands=bands, res=512, show=False)
        plot.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'spectral-topo', population, f'{population}-spectral-topo-{bin_}.png'))
        
        print(f'====================== spectral topo plotted for {population} - {bin_}')

def plot_psd_single_subj(subject_id, input_dir, output_dir):

    print(f'====================== plotting psd for {subject_id}')
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-plots')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-plots'))
        print(f'====================== psd plots dir created for {subject_id}')

    spectrum_dict = load_psd_single_subj(subject_id, input_dir)

    for bin_, spectrum in spectrum_dict.items():
        fig, ax = plt.subplots()
        spectrum.plot(average=True, dB=False, ci_alpha=0.2, show=False, axes=ax)
        ax.set_ylabel('Power')
        ax.set_title(f'sub-{subject_id} - {bin_}')
        fig.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-plots', f'sub-{subject_id}-psd-{bin_}.png'))
        plt.close()
        print(f'====================== psd saved for {subject_id} - {bin_}')

    fig, ax = plt.subplots()
    spectrum_dict['dis_mid'].plot(average=True, dB=False, ci_alpha=0.1,  show=False, axes=ax, color='red')
    spectrum_dict['dis_lat'].plot(average=True, dB=False, ci_alpha=0.1, show=False, axes=ax, color='blue')
    spectrum_dict['no_dis'].plot(average=True, dB=False, ci_alpha=0.1, show=False, axes=ax, color='green')
    ax.set_title(f'sub-{subject_id} - dis_mid vs dis_lat vs no_dis')
    ax.set_ylabel('Power')
    fig.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'psd', 'psd-plots', f'sub-{subject_id}-psd-3conds.png'))
    plt.close()


def plot_psd_population(input_dir, output_dir, subject_list, population):

    spectrum_dict = load_psd_population(input_dir, output_dir, subject_list, population)

    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'psd', 'psd-plots', population)):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'psd', 'psd-plots', population))
        print(f'====================== psd plots dir created for {population}')

    for bin_, spectrum in spectrum_dict.items():
        fig, ax = plt.subplots()
        spectrum.plot(average=True, dB=False, ci_alpha=0.2, show=False, axes=ax)
        ax.set_ylabel('Power')
        ax.set_title(f'{population} - {bin_}')
        fig.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'psd', 'psd-plots', population, f'{population}-psd-{bin_}.png'))
        plt.close()
        print(f'====================== psd saved for {population} - {bin_}')

    fig, ax = plt.subplots()
    spectrum_dict['dis_mid'].plot(average=True, dB=False, ci_alpha=0.1,  show=False, axes=ax, color='red')
    spectrum_dict['dis_lat'].plot(average=True, dB=False, ci_alpha=0.1, show=False, axes=ax, color='blue')
    spectrum_dict['no_dis'].plot(average=True, dB=False, ci_alpha=0.1, show=False, axes=ax, color='green')
    ax.set_title(f'{population} - dis_mid vs dis_lat vs no_dis')
    ax.set_ylabel('Power')
    fig.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'psd', 'psd-plots', population, f'{population}-psd-3conds.png'))
    plt.close()
    

### ================================================================================================
### ==================================== N2PC ALPHA POWER PER EPOCH ================================
### ============================================== FFT =============================================
### ================================================================================================

# 3rd version of the functions. This time we compute the alpha power for each individual electrode using FFT.
# For that we use the mne.time_frequency.Spectrum class. 

def get_psd(subject_id, input_dir):

    # load the epochs
    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs',f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))
    # clear bad channels
    epochs.info['bads'] = []
    # create spectrum object
    spec = epochs.compute_psd(fmin=0, fmax=35)
    print(f'========================= PSD for subject {subject_id} computed!')

    return spec

def get_mean_freq(spec, bands, picks, epoch_index):

    # Get the equivalent indices of the bands in the data (from .get_data() method)
    lower_bound = bands[0]*2
    upper_bound = bands[1]*2+1

    # Compute the mean of the epochs for each freq
    spec_mean_power = spec.get_data(picks=picks)[epoch_index:epoch_index+1, :, lower_bound:upper_bound].mean(axis=2)
    print(f'========================= Mean power for {picks} computed!')

    return spec_mean_power

def get_power_df_single_subj(subject_id, input_dir, output_dir):


    # load the epochs
    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs',f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))

    # Get the spec object
    spec = get_psd(subject_id, input_dir)

    # Define the frequency bands
    bands = {'theta' : np.arange(4, 9),
         'alpha' : np.arange(8, 13),
         'low_beta' : np.arange(12, 17),
         'high_beta' : np.arange(16, 31)
    
    }

    # Define the picks
    channels = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7',
      'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4','F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4',
        'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

    # Define the columns of the dataframe : subject ID, epoch index, epoch dropped, reset index, condition, target side,
    # distractor position, theta Fp1, alpha Fp1, low beta Fp1, high beta Fp1, theta AF7, alpha AF7, low beta AF7, high beta AF7, etc...
    ch_list = []
    for ch in channels:
        for band in ['theta', 'alpha', 'low_beta', 'high_beta']:
            ch_list.append(f'{band}-{ch}')
    columns = ['ID', 'epoch_index', 'epoch_dropped', 'index_reset','saccade','condition', 'target_side', 'distractor_position'] + ch_list

    # Create the dataframe
    df = pd.DataFrame(columns=columns)

    # Populate the df
    # Start by getting the epochs indices, the epochs status (dropped or not) and the reset indices

    # Load the reject log
    reject_log = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'preprocessing', 'step-06-final-reject_log', f'sub-{subject_id}-final-reject_log-N2pc.npz'))
    # Define the epochs status (rejected or not)
    epochs_status = reject_log['bad_epochs']
    
    # Create row for each epoch
    df['ID'] = [subject_id] * len(epochs_status)
    df['epoch_index'] = range(1,len(epochs_status)+1)
    df['epoch_dropped'] = epochs_status
    
    # Add a column that store the reset index of the epochs
    index_val = 0
    index_list = []

    # Iterate through the 'epoch_dropped' column to create the reset index column
    for row_number in range(len(df)):
        if df.iloc[row_number, 2] == False:
            index_list.append(index_val)
            index_val += 1
        else:
            index_list.append(np.nan)

    # Add the reset index column to the DataFrame
    df['index_reset'] = index_list

    # Load the csv file contaning the indices of epochs with saccades
    saccades = pd.read_csv(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'rejected-epochs-list', f'sub-{subject_id}-heog-artifact.csv'))
    # Create a list of the indices of the epochs with saccades
    saccades_list = list(saccades['index'])
    # Add a column that specifies if the epoch contains a saccade. FALSE if no saccade, TRUE if saccade. 
    df['saccade'] = df['index_reset'].isin(saccades_list)

    # Fill the condition and the power columns
    for row_number in range(len(df)):

        if df.iloc[row_number, 2] == False:

            # add condition
            df.iloc[row_number, 5] = epochs.events[int(df['index_reset'].loc[row_number]),2]

            # add target side
            if df.iloc[row_number, 5] % 2 == 0:
                df.iloc[row_number, 6] = 'right'
            elif df.iloc[row_number, 5] % 2 != 0:
                df.iloc[row_number,6] = 'left'

            # add dis position
            if df.iloc[row_number, 5] in [1,2,5,6]:
                df.iloc[row_number, 7] = 'mid'
            elif df.iloc[row_number, 5] in [3,4]:
                df.iloc[row_number, 7] = 'nodis'
            elif df.iloc[row_number, 5] in [7,8]:
                df.iloc[row_number, 7] = 'lat'
            
            # Compute the mean power for each band and each electrode
            for i, ch in enumerate(ch_list):
                band, pick = ch.split('-')
                power = get_mean_freq(spec, bands=bands[band], picks=pick, epoch_index=int(df['index_reset'].loc[row_number]))
                df.iloc[row_number, i+8] = power[0][0] # +8 because the first 8 columns are not electrodes

    print(f'========================= Dataframe created for subject {subject_id}!')
    # save the dataframe
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'alpha-power-df')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}','N2pc', 'alpha-power-df'))
    df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'alpha-power-df', f'sub-{subject_id}-alpha-power-per-epoch_v2.csv'), index=False)
    print(f'========================= Dataframe saved for subject {subject_id}!')

    return df

def get_power_df_all_subj(input_dir, output_dir):

    # Get the list of all the subject directories
    subject_list = [os.path.basename(subj) for subj in glob.glob(os.path.join(input_dir, 'sub-*'))]
    # Sort the list
    subject_list.sort()

    # Create a list to store the dataframes
    df_list = []

    # Loop over the subject directories
    for subject in subject_list:

        subject_id = subject[-2:]
        # Compute alpha power and save it in a dataframe
        try:
            df = get_power_df_single_subj(subject_id, input_dir, output_dir)
            df_list.append(df)
            print(f'==================== Dataframe created and saved for subject {subject_id}! :)')
        except:
            print(f"==================== No data (epochs or reject log) for subject {subject_id}! O_o'")
            continue
    
    # Concatenate all dataframes in the list
    big_df = pd.concat(df_list)

    # Save dataframe as .csv file
    if not os.path.exists(os.path.join(output_dir,'all_subj','N2pc', 'alpha-power-allsubj')):
        os.makedirs(os.path.join(output_dir,'all_subj','N2pc', 'alpha-power-allsubj'))
    big_df.to_csv(os.path.join(output_dir,'all_subj','N2pc', 'alpha-power-allsubj', 'alpha-power-per-epoch-allsubj_v2.csv'), index=False)

    return big_df

### ================================================================================================
### ==================================== N2PC ALPHA POWER PER EPOCH ================================
### ============================================== CWT =============================================
### ================================================================================================


def alpha_power_per_epoch(subject_id, input_dir, right_elecs=['O2', 'PO4', 'PO8'], left_elecs=['O1', 'PO3', 'PO7']):
    ''' Compute the alpha power for each epoch and return a list of alpha power values for the right side of the head and a list of alpha power values for the left side of the head.

    Parameters:
    ----------
    epochs : mne.Epochs
        The epochs object to be sorted.
    
    Returns
    ----------
    right_power_list : list of float
        a list of alpha band mean power values (8-12Hz) for the right side of the head for each epoch
    
    left_power_list : list of float
        a list of alpha band mean power values (8-12Hz) for the left side of the head for each epoch
    '''

    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs',f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))

    freqs = np.arange(8, 13)

    n_cycles = freqs / 2.
    time_bandwidth = 4.
    n_jobs = 1  # number of parallel jobs to run
    
    right_power_list = []
    left_power_list = []

    print('========================= Computing alpha power, might take a while...')

    for i in range(len(epochs)):

        right_power = mne.time_frequency.tfr_morlet(epochs[i], freqs=freqs, n_cycles=n_cycles, picks=right_elecs,
                                                        use_fft=True, return_itc=False, decim=1,
                                                        n_jobs=n_jobs, verbose=True)
        right_power_mean = right_power.to_data_frame()[right_elecs].mean(axis=1).mean(axis=0)
        right_power_list.append(right_power_mean)

        left_power= mne.time_frequency.tfr_morlet(epochs[i], freqs=freqs, n_cycles=n_cycles, picks=left_elecs,
                                                        use_fft=True, return_itc=False, decim=1,
                                                        n_jobs=n_jobs, verbose=True)
        left_power_mean = left_power.to_data_frame()[left_elecs].mean(axis=1).mean(axis=0)
        left_power_list.append(left_power_mean)

    return right_power_list, left_power_list

def alpha_df_epoch(subject_id : str, input_dir, right_power_list, left_power_list):
    ''' Create a dataframe with the alpha power for each epoch

    Parameters:
    ----------
    epochs : mne.Epochs
        The epochs object to be sorted.

    right_power_list : list of float
        a list of alpha band mean power values (8-12Hz) for the right side of the head for each epoch

    left_power_list : list of float
        a list of alpha band mean power values (8-12Hz) for the left side of the head for each epoch

    Returns
    ----------
    df : pandas.DataFrame
        A dataframe with the alpha power score for each side for each epoch

    '''
    # Load the epochs
    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))
    # Load the reject log
    reject_log = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'preprocessing', 'step-06-final-reject_log', f'sub-{subject_id}-final-reject_log-N2pc.npz'))
    # Define the epochs status (rejected or not)
    epochs_status = reject_log['bad_epochs']

    # Initiate the df
    df = pd.DataFrame(columns=[['epoch_index', 'epoch_dropped', 'condition','target_side', 'distractor_position', 'alpha_power_right', 'alpha_power_left']])
    
    # Create row for each epoch
    df['epoch_index'] = range(1,len(epochs_status)+1)
    df['epoch_dropped'] = epochs_status

    # Add aa column that store the reset index of the epochs
    index_val = 0
    index_list = []

    # Iterate through the 'epoch_dropped' column to create the reset index column
    for row_number in range(len(df)):
        if df.iloc[row_number, 1] == False:
            index_list.append(index_val)
            index_val += 1
        else:
            index_list.append(np.nan)
    # Add the index column to the DataFrame
    df['index_reset'] = index_list
    
    for row_number in range(len(df)):

        if df.iloc[row_number, 1] == False:

            # add condition
            df.iloc[row_number, 2] = epochs.events[int(df['index_reset'].loc[row_number]),2]

            # add target side
            if df.iloc[row_number, 2] % 2 == 0:
                df.iloc[row_number, 3] = 'right'
            elif df.iloc[row_number, 2] % 2 != 0:
                df.iloc[row_number,3] = 'left'

            # add dis position
            if df.iloc[row_number, 2] in [1,2,5,6]:
                df.iloc[row_number, 4] = 'mid'
            elif df.iloc[row_number, 2] in [3,4]:
                df.iloc[row_number, 4] = 'nodis'
            elif df.iloc[row_number, 2] in [7,8]:
                df.iloc[row_number, 4] = 'lat'
                
            # add alpha power right
            df.iloc[row_number, 5] = right_power_list[int(df['index_reset'].loc[row_number])]
            # add alpha power left
            df.iloc[row_number, 6] = left_power_list[int(df['index_reset'].loc[row_number])]
    
    # Scientific notification because very small values
    pd.options.display.float_format = '{:.5e}'.format

    return df    

def alpha_df_epoch_3clusters(subject_id, input_dir):

    # get the alpha power for each epoch for each cluster
    cluster_dict = {'occipital': {'right':['O2', 'PO4', 'PO8'], 'left': ['O1', 'PO3', 'PO7']},
                    'parietal' : {'right':['P2', 'CP2', 'CP4'], 'left':['P1', 'CP1', 'CP3']}, 'frontal':{'right':['FC2', 'FC4', 'F2'], 'left':['FC1', 'FC3', 'F1']}}


    right_occip, left_occip = alpha_power_per_epoch(subject_id, input_dir, right_elecs=cluster_dict['occipital']['right'], left_elecs=cluster_dict['occipital']['left'])
    right_parietal, left_parietal = alpha_power_per_epoch(subject_id, input_dir, right_elecs=cluster_dict['parietal']['right'], left_elecs=cluster_dict['parietal']['left'])
    right_frontal, left_frontal = alpha_power_per_epoch(subject_id, input_dir, right_elecs=cluster_dict['frontal']['right'], left_elecs=cluster_dict['frontal']['left'])

    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))
    if type(epochs) == mne.epochs.EpochsFIF:
        print(f'=========== epochs found for subject {subject_id}')
    else:
        print(f'=========== epochs not found for subject {subject_id}')
    # Load the reject log
    reject_log = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'preprocessing', 'step-06-final-reject_log', f'sub-{subject_id}-final-reject_log-N2pc.npz'))
    # Define the epochs status (rejected or not)
    epochs_status = reject_log['bad_epochs']

    # Initiate the df
    df = pd.DataFrame(columns=[['ID', 'epoch_index', 'epoch_dropped', 'condition','target_side', 'distractor_position', 'alpha_power_right_occip', 'alpha_power_left_occip',
                                'alpha_power_right_parietal', 'alpha_power_left_parietal', 'alpha_power_right_frontal', 'alpha_power_left_frontal']])
    
    # Create row for each epoch
    df['ID'] = [subject_id] * len(epochs_status)
    df['epoch_index'] = range(1,len(epochs_status)+1)
    df['epoch_dropped'] = epochs_status

    # Add aa column that store the reset index of the epochs
    index_val = 0
    index_list = []

    # Iterate through the 'epoch_dropped' column to create the reset index column
    for row_number in range(len(df)):
        if df.iloc[row_number, 2] == False:
            index_list.append(index_val)
            index_val += 1
        else:
            index_list.append(np.nan)
    # Add the index column to the DataFrame
    df['index_reset'] = index_list

    for row_number in range(len(df)):

        if df.iloc[row_number,2] == False:

            # add condition
            df.iloc[row_number, 3] = epochs.events[int(df['index_reset'].loc[row_number]),2]

            # add target side
            if df.iloc[row_number, 3] % 2 == 0:
                df.iloc[row_number, 4] = 'right'
            elif df.iloc[row_number, 3] % 2 != 0:
                df.iloc[row_number,4] = 'left'

            # add dis position
            if df.iloc[row_number, 3] in [1,2,5,6]:
                df.iloc[row_number, 5] = 'mid'
            elif df.iloc[row_number, 3] in [3,4]:
                df.iloc[row_number, 5] = 'nodis'
            elif df.iloc[row_number, 3] in [7,8]:
                df.iloc[row_number, 5] = 'lat'
                
            # add alpha power right occipital
            df.iloc[row_number, 6] = right_occip[int(df['index_reset'].loc[row_number])]
            # add alpha power left occipital
            df.iloc[row_number, 7] = left_occip[int(df['index_reset'].loc[row_number])]
            # add alpha power right parietal
            df.iloc[row_number, 8] = right_parietal[int(df['index_reset'].loc[row_number])]
            # add alpha power left parietal
            df.iloc[row_number, 9] = left_parietal[int(df['index_reset'].loc[row_number])]
            # add alpha power right frontal
            df.iloc[row_number, 10] = right_frontal[int(df['index_reset'].loc[row_number])]
            # add alpha power left frontal
            df.iloc[row_number, 11] = left_frontal[int(df['index_reset'].loc[row_number])]
    
    # Scientific notification because very small values
    pd.options.display.float_format = '{:.5e}'.format

    return df

def single_subj_alpha_epoch(subject_id, input_dir, output_dir):

    subject_id = str(subject_id)

    df = alpha_df_epoch_3clusters(subject_id, input_dir)

    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'alpha-power-df')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}','N2pc', 'alpha-power-df'))
    df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'alpha-power-df', f'sub-{subject_id}-alpha-power-per-epoch.csv'))

    return df
    
# ======================================== ALL SUBJECTS ==========================================

def all_subj_alpha_epoch(input_dir, output_dir):
    ''' Create a dataframe for all subjects with the alpha score for each side and for each epoch
    
    Parameters:
    ----------
    Input_dir : str
        The path to the directory containing the epochs objects for each subject
    output_dir : str
        The path to the directory where the dataframe will be saved

    Returns
    ----------
    big_df : pandas.DataFrame
        A dataframe with the alpha power score for each side for each subject
    '''
    print('========================= WARNING')
    print('THIS FUNCTION SHOULD NOT BE USED, IT DOES NOT TRACK THE DROPPED EPOCHS')
    print('USE THE FUNCTION alpha_power_per_epoch INSTEAD - SEE THE FILE compute_alpha_epoch.py')
    print('========================= WARNING')

    big_df = pd.DataFrame()

    # Empty list to store the files
    all_subj_files = []

    # Loop over the directories to access the files
    directories = glob.glob(os.path.join(input_dir, 'sub*', 'N2pc'))
    for directory in directories:
        file = glob.glob(os.path.join(directory, 'cleaned_epochs', 'sub*N2pc.fif'))
        all_subj_files.append(file[0])
    all_subj_files.sort()

    # Pattern to extract the subject ID
    pattern = r'sub-(\d{2})-cleaned'

    for subject in all_subj_files:

        # Extract the subject ID
        match = re.search(pattern, subject)
        if match:
            subj_id = match.group(1)
        else:
            print("No match found in:", subject)

        print(f'========================= working on {subj_id}')

        # Load the epochs
        right, left = alpha_power_per_epoch(subject_id=subj_id, input_dir=input_dir)
        subj_df = alpha_df_epoch(subject_id=subj_id, input_dir=input_dir, right_power_list=right, left_power_list=left)

        subj_df.insert(0, 'ID', subj_id)

        big_df = pd.concat([big_df, subj_df], ignore_index=True)
    
    if not os.path.exists(os.path.join(output_dir, 'all_subj' 'alpha-power-df')):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'alpha-power-df'))
    big_df.to_csv(os.path.join(output_dir, 'all-subj', 'alpha-power-df', 'allsubj_alpha_power_per_epoch.csv'))

    return big_df


