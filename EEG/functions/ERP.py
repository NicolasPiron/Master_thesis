import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import math
from functions.file_management import add_sub0


def to_evoked(subject_id, task, input_dir):
    ''' This function converts the epochs to evoked objects and saves them in the subject directory.
        It saves one evoked file by condition (i.e. bin).

    Parameters
    ----------
    subject_id : str
        The subject ID to plot.
    task : str
        The task to plot.
    input_dir : str
        The path to the directory containing the input data.
    
    Returns
    -------
    None
    
    '''
    # load the epochs
    file = os.path.join(input_dir, f'sub-{subject_id}/cleaned_epochs/sub-{subject_id}-cleaned_epochs-{task}.fif')
    epochs = mne.read_epochs(file)

    if task == 'N2pc':

        # crop the epochs to the relevant time window
        tmin = -0.2
        tmax = 0.4
        epochs.crop(tmin=tmin, tmax=tmax)
            
        # define the bins
        bins = {'bin1' : ['dis_top/target_l','dis_bot/target_l'],
                'bin2' : ['dis_top/target_r','dis_bot/target_r'],
                'bin3' : ['no_dis/target_l'],
                'bin4' : ['no_dis/target_r'],
                'bin5' : ['dis_right/target_l'],
                'bin6' : ['dis_left/target_r']}

        # create evoked
        evoked_list = [epochs[bin].average() for bin in bins.values()]
        
        # rename the distractor mid conditions to simplify
        evoked_1 = evoked_list[0]
        evoked_2 = evoked_list[1]
        evoked_1.comment = 'dis_mid/target_l'
        evoked_2.comment = 'dis_mid/target_r'
        
        # replace the '/' that causes problems when saving
        for evoked in evoked_list:
            evoked.comment = evoked.comment.replace('/', '_')
        

        # save the evoked objects in subject directory
        if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', f'evoked-{task}')):
            os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', f'evoked-{task}'))
        for evoked in evoked_list:
            print(evoked.comment)
            evoked.save(os.path.join(input_dir, f'sub-{subject_id}', f'evoked-{task}', f'sub-{subject_id}-{evoked.comment}-ave.fif'), overwrite=True)



def get_evoked(subject_id, input_dir):
    ''' This function loads the evoked files for a given subject and returns a dictionary
    
    Parameters
    ----------
    subject_id : str
        The subject ID to plot.
    input_dir : str
        The path to the directory containing the input data.
    
    Returns
    -------
    bin_evoked : dict
        A dictionary containing the evoked objects for each condition (i.e. bin).
    '''

    subject_id = str(subject_id)
    evoked_path = os.path.join(input_dir, f'sub-{subject_id}', f'evoked-N2pc')
    evoked_files = glob.glob(os.path.join(evoked_path, f'sub-{subject_id}-*.fif'))
    # Load the evoked files
    evoked_list = [mne.read_evokeds(evoked_file)[0] for evoked_file in evoked_files]
    bin_dict = {'bin1' : 'dis_mid_target_l',
            'bin2' : 'dis_mid_target_r',
            'bin3' : 'no_dis_target_l',
            'bin4' : 'no_dis_target_r',
            'bin5' : 'dis_right_target_l',
            'bin6' : 'dis_left_target_r'}

    # Assign the evoked object that corresponds to the bin
    bin_evoked = {}

    for bin_name, comment in bin_dict.items():
        for evoked in evoked_list:
            if evoked.comment == comment:
                bin_evoked[bin_name] = evoked
                break 

    # Rename the keys of the dict
    prefix = 'evk_'
    # Create a new dictionary with modified keys
    bin_evoked = {prefix + key: value for key, value in bin_evoked.items()}

    return bin_evoked

def get_non_excluded_subjects_list(excluded_subjects_list, input_dir, exclude_subjects=False):
    ''' Gives you a list of all the subjects in the input_dir, excluding the subjects in the excluded_subjects_list.

    Parameters
    ----------
    excluded_subjects_list : list
        List of subjects to be excluded from the grand average.
    input_dir : str
        The path to the directory containing the input data.
    exclude_subjects : bool
        Whether to exclude subjects or not
    
    Returns
    -------
    subject_list : list
        List of all the subjects in the input_dir, excluding the subjects in the excluded_subjects_list.
    '''
    if exclude_subjects == True:
        
        # transform the excluded_subjects_list to match the format of the subject IDs
        transformed_list = add_sub0(excluded_subjects_list)
        print(transformed_list)

        # get the list of all subjects
        subject_list = glob.glob(os.path.join(input_dir, 'sub*'))

        # exclude the subjects in the excluded_subjects_list
        subjects_to_keep = []
        for subject in subject_list:
            print(subject[-6:])
            if subject[-6:] in transformed_list:
                print(f'====================== Excluding {subject}')
            else:
                subjects_to_keep.append(subject)
        subject_list = subjects_to_keep

    else:
        subject_list = glob.glob(os.path.join(input_dir, 'sub*'))
    
    return subject_list


def combine_evoked(subject_id, input_dir, output_dir, exclude_subjects=False, excluded_subjects_list=[], population=None):
    ''' This function concatenates the evoked files into 3 conditions
        for a given subject and saves them in the subject directory.
        It aslo combine subjects into a grand average if subject_id = 'GA'.
    
    Parameters
    ----------
    subject_id : str
        The subject ID to plot. Can be 'GA' to plot the grand average.
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.
    exclude_subjects : bool
        Whether to exclude subjects from the grand average.
    excluded_subjects_list : list
        List of subjects to be excluded from the grand average.
    population : str
        Population (control or stroke).

    Returns
    -------
    None
    '''

    def combine_evoked_single_subj(subject_id, input_dir, output_dir):  
        # load the evoked files
        bin_evoked = get_evoked(subject_id, input_dir)

        print(bin_evoked)

        # create to list of evoked objects for each condition
        dis_mid = [bin_evoked[bin_].crop(tmin=0, tmax=0.4) for bin_ in ['evk_bin1', 'evk_bin2']]
        no_dis = [bin_evoked[bin_].crop(tmin=0, tmax=0.4) for bin_ in ['evk_bin3', 'evk_bin4']]
        dis_contra = [bin_evoked[bin_].crop(tmin=0, tmax=0.4) for bin_ in ['evk_bin5', 'evk_bin6']]

        # find right and left channels
        ch_names = list(bin_evoked.values())[0].info['ch_names']
        LCh = []
        RCh = []
        for i, ch in enumerate(ch_names):
            if str(ch[-1]) == 'z':
                print(f'central channel {ch} -> not included in lateral channels list')
            elif int(ch[-1]) % 2 == 0:
                RCh.append(i)
            elif int(ch[-1]) %2 != 2:
                LCh.append(i) 

        # combine the evoked objects
        pairs = [dis_mid, no_dis, dis_contra]
        pair_names = ['dis_mid', 'no_dis', 'dis_contra']
        for i, pair in enumerate(pairs):
            # the right target evoked object will be laterally swapped so it is like the target is on the left
            to_swap = pair[1]
            data = to_swap.get_data()
            swapped_data = data.copy()
            swapped_data[RCh] = data[LCh]
            swapped_data[LCh] = data[RCh]
            swapped = mne.EvokedArray(swapped_data, to_swap.info)
            combined_pair = mne.combine_evoked([pair[0], swapped], weights='equal')
            combined_pair.comment = pair_names[i]
            # save the combined evoked object
            if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', f'evoked-N2pc', 'combined')):
                os.makedirs(os.path.join(output_dir ,f'sub-{subject_id}', f'evoked-N2pc', 'combined'))
            combined_pair.save(os.path.join(output_dir, f'sub-{subject_id}', f'evoked-N2pc', 'combined', f'sub-{subject_id}-{pair_names[i]}-ave.fif'), overwrite=True)


    subject_id = str(subject_id)

    if subject_id == 'GA':
    
        subject_list = get_non_excluded_subjects_list(excluded_subjects_list, input_dir, exclude_subjects=exclude_subjects)

        if population == 'control':

            # keep only the control subjects (i.e. subject IDs < 50)
            subject_list = [subject for subject in subject_list if int(subject[-2:]) < 50]
            print(f'====================== subjects list, control population : {subject_list}')
            

        elif population == 'stroke':

            # keep only the stroke subjects (i.e. subject IDs > 50)
            subject_list = [subject for subject in subject_list if int(subject[-2:]) > 50]
            print(f'====================== subjects list, stroke population : {subject_list}')


        # create lists of evoked objects for each condition
        dis_mid_list = []
        no_dis_list = []
        dis_contra_list = []

        for sub in subject_list:
            subject_id = str(sub[-2:])
            # check is the combined evoked files already exist
            if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'evoked-N2pc', 'combined', f'sub-{subject_id}-dis_mid-ave.fif')):
                
                combine_evoked_single_subj(subject_id, input_dir, output_dir)
                print(f'====================== evoked files combined for {subject_id}')
            else:
                print(f'====================== evoked files were already combined for {subject_id}')
            
            # loop over the subjects and append the evoked objects to the lists
            evoked_files = glob.glob(os.path.join(input_dir, f'sub-{subject_id}', f'evoked-N2pc', 'combined', f'sub-{subject_id}-*ave.fif'))
            evoked_list = [mne.read_evokeds(evoked_file)[0] for evoked_file in evoked_files]
            evoked_dict = {}
            for evoked in evoked_list:
                evoked_dict[evoked.comment] = evoked
            dis_mid_list.append(evoked_dict['dis_mid'])
            no_dis_list.append(evoked_dict['no_dis'])
            dis_contra_list.append(evoked_dict['dis_contra'])
        
        # combine the evoked objects
        dis_mid_combined = mne.combine_evoked(dis_mid_list, weights='equal')
        no_dis_combined = mne.combine_evoked(no_dis_list, weights='equal')
        dis_contra_combined = mne.combine_evoked(dis_contra_list, weights='equal')
        dis_mid_combined.comment = 'dis_mid'
        no_dis_combined.comment = 'no_dis'
        dis_contra_combined.comment = 'dis_contra'

        # save the combined evoked objects
        if not os.path.exists(os.path.join(output_dir, 'all_subj', 'evoked-N2pc', 'combined', population)):
            os.makedirs(os.path.join(output_dir, 'all_subj', 'evoked-N2pc', 'combined', population))
        
        if population == 'control':
            str_excluded_subjects_list = [str(sub) for sub in excluded_subjects_list if sub < 50]
            excluded_subjects_string = '_'.join(str_excluded_subjects_list)
        elif population == 'stroke':
            str_excluded_subjects_list = [str(sub) for sub in excluded_subjects_list if sub > 50]
            excluded_subjects_string = '_'.join(str_excluded_subjects_list)

        if exclude_subjects == True:
            file_name_dis_mid = f'{population}-excluded_subjects-{excluded_subjects_string}-dis_mid-ave.fif'
            file_name_no_dis = f'{population}-excluded_subjects-{excluded_subjects_string}-no_dis-ave.fif'
            file_name_dis_contra = f'{population}-excluded_subjects-{excluded_subjects_string}-dis_contra-ave.fif'
        else:
            file_name_dis_mid = f'{population}-dis_mid-ave.fif'
            file_name_no_dis = f'{population}-no_dis-ave.fif'
            file_name_dis_contra = f'{population}-dis_contra-ave.fif'
        
        dis_mid_combined.save(os.path.join(output_dir, 'all_subj', 'evoked-N2pc', 'combined', population, file_name_dis_mid), overwrite=True)
        no_dis_combined.save(os.path.join(output_dir, 'all_subj', 'evoked-N2pc', 'combined', population, file_name_no_dis), overwrite=True)
        dis_contra_combined.save(os.path.join(output_dir, 'all_subj', 'evoked-N2pc', 'combined', population, file_name_dis_contra), overwrite=True)
            
    else:

        combine_evoked_single_subj(subject_id, input_dir, output_dir)


def plot_erp_topo(subject_id, input_dir, output_dir, exclude_subjects=False, excluded_subjects_list=[], population=None):
    '''
    Parameters
    ----------
    subject_id : str
        The subject ID to plot. Can be 'GA' to plot the grand average.
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.
    exclude_subjects : bool
        Whether to exclude subjects from the grand average.
    excluded_subjects_list : list
        List of subjects to be excluded from the grand average.
    population : str
        Population (control or stroke).

    Returns
    -------
    None
    '''

    subject_id = str(subject_id)

    def load_combined_evoked(evoked_list):
        evoked_dict = {}
        for evoked in evoked_list:
            evoked_dict[evoked.comment] = evoked
        return evoked_dict
    
    # define the adequate path to the data and the output directory
    if subject_id == 'GA':
        print('====================== plotting for GA')
        # input data
        evoked_combined_path = os.path.join(input_dir, 'all_subj', 'evoked-N2pc', 'combined', population)

        # output directory
        out_path = os.path.join(output_dir, 'all_subj', 'n2pc-plots', population, 'n2pc-topo')
        
        if exclude_subjects == True:
            print(f'====================== excluding subjects {excluded_subjects_list}')
            excluded_subjects_string = '_'.join([str(sub) for sub in excluded_subjects_list])
            # important : file_name_start is used to find the files and save the plots
            file_name_start = f'{population}-excluded_subjects-{excluded_subjects_string}'

        if exclude_subjects == False:
            print(f'====================== not excluding subjects')
            file_name_start = f'{population}'
    else:
        print(f'====================== plotting for {subject_id}')
        # input data
        evoked_combined_path = os.path.join(input_dir, f'sub-{subject_id}', 'evoked-N2pc', 'combined')
        file_name_start = f'sub-{subject_id}'

        # output directory
        out_path = os.path.join(output_dir, f'sub-{subject_id}', 'n2pc-plots', 'n2pc-topo')

    # load data and store it in a dictionary
    evoked_combined_files = glob.glob(os.path.join(evoked_combined_path, f'{file_name_start}*.fif'))
    if len(evoked_combined_files) == 0:
        print('====================== no file found')
    evoked_list = [mne.read_evokeds(evoked_file)[0] for evoked_file in evoked_combined_files]
    evoked_dict = load_combined_evoked(evoked_list)
    # plot the topomaps
    for bin_, evoked in evoked_dict.items():
        topo = evoked.plot_topomap(times=[0.1, 0.15, 0.2, 0.25, 0.3], show=False)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        bin_name = bin_.split(' ')[-1]
        topo.savefig(os.path.join(out_path, f'{file_name_start}-topo-{bin_name}.png'))


def plot_spectral_topo(subject_id, input_dir, output_dir, exclude_subjects=False, excluded_subjects_list=[], population=None):
    '''
    
    Parameters
    ----------
    subject_id : str
        The subject ID to plot. Can be 'GA' to plot the grand average.
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.
    exclude_subjects : bool
        Whether to exclude subjects from the grand average.
    excluded_subjects_list : list
        List of subjects to be excluded from the grand average.
    population : str
        Population (control or stroke).

    Returns
    -------
    None
    '''

    subject_id = str(subject_id)

    def load_combined_evoked(evoked_list):
        evoked_dict = {}
        for evoked in evoked_list:
            evoked_dict[evoked.comment] = evoked
        return evoked_dict
    
    # define the adequate path to the data and the output directory
    if subject_id == 'GA':
        print('====================== plotting for GA')
        # input data
        evoked_combined_path = os.path.join(input_dir, 'all_subj', 'evoked-N2pc', 'combined', population)

        # output directory
        out_path = os.path.join(output_dir, 'all_subj', 'spectral-topo', population)
        
        if exclude_subjects == True:
            print(f'====================== excluding subjects {excluded_subjects_list}')
            excluded_subjects_string = '_'.join([str(sub) for sub in excluded_subjects_list])
            file_name_start = f'{population}-excluded_subjects-{excluded_subjects_string}'
        if exclude_subjects == False:
            print(f'====================== not excluding subjects')
            file_name_start = f'{population}'
    else:
        print(f'====================== plotting for {subject_id}')
        # input data
        evoked_combined_path = os.path.join(input_dir, f'sub-{subject_id}', 'evoked-N2pc', 'combined')
        file_name_start = f'sub-{subject_id}'

        # output directory
        out_path = os.path.join(output_dir, f'sub-{subject_id}', 'spectral-topo')

    # load data and store it in a dictionary
    evoked_combined_files = glob.glob(os.path.join(evoked_combined_path, f'{file_name_start}*.fif'))
    if len(evoked_combined_files) == 0:
        print('====================== no file found')
    evoked_list = [mne.read_evokeds(evoked_file)[0] for evoked_file in evoked_combined_files]
    evoked_dict = load_combined_evoked(evoked_list)
    
    # transform the evoked objects into spectrum objects
    spectrum_dict = {}
    for key, value in evoked_dict.items():
        spectrum_dict[key] = value.compute_psd()
    
    # plot the spectrum for alpha and beta frequency bands
    bands = {'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30)}

    for bin_, spectrum in spectrum_dict.items():    
        plot = spectrum.plot_topomap(bands=bands, vlim='joint', res=512, show=False)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        bin_name = bin_.split(' ')[-1]
        plot.savefig(os.path.join(out_path, f'{file_name_start}-spectral-topo-{bin_name}.png'))


def get_bins_data(subject_id, input_dir, exclude_subjects=False, excluded_subjects_list=[], population=None):
    ''' This function extracts the N2pc ERP values for a given subject, or all of them
        if subject_id = 'GA'.

    Parameters
    ----------
    subject_id : str
        The subject ID OR 'GA' to get the grand average.
    input_dir : str
        The path to the directory containing the input data.
    exclude_subjects : bool
        Whether to exclude subjects from the grand average.
    excluded_subjects_list : list
        List of subjects to be excluded from the grand average.
    population : str
        Population (control or stroke).
    
    Returns
    -------
    PO7_data_nbin1 : numpy.ndarray
        The N2pc ERP data for the Dis_Mid contra condition.
    PO7_data_nbin2 : numpy.ndarray
        The N2pc ERP data for the Dis_Mid ipsi condition.
    PO7_data_nbin3 : numpy.ndarray
        The N2pc ERP data for the No_Dis contra condition.
    PO7_data_nbin4: numpy.ndarray
        The N2pc ERP data for the No_Dis ipsi condition.
    PO7_data_nbin5 : numpy.ndarray
        The N2pc ERP data for the Dis_Contra contra condition.
    PO7_data_nbin6 : numpy.ndarray
        The N2pc ERP data for the Dis_Contra ipsi condition.
    time : numpy.ndarray
        The time axis for the ERP data.
    '''
    def get_evoked_data(subject_id, input_dir):

        bin_evoked = get_evoked(subject_id, input_dir)
        
        # Define the channel indices for left (Lch) and right (Rch) channels
        Lch = np.concatenate([np.arange(0, 27)])
        Rch = np.concatenate([np.arange(33, 36), np.arange(38, 46), np.arange(48, 64)])

        # Lateralize bins in order to be able to compute the N2pc
        evk_bin1_R = bin_evoked['evk_bin1'].copy().pick(Rch)
        evk_bin1_L = bin_evoked['evk_bin1'].copy().pick(Lch)
        evk_bin2_R = bin_evoked['evk_bin2'].copy().pick(Rch)
        evk_bin2_L = bin_evoked['evk_bin2'].copy().pick(Lch)
        evk_bin3_R = bin_evoked['evk_bin3'].copy().pick(Rch)
        evk_bin3_L = bin_evoked['evk_bin3'].copy().pick(Lch)
        evk_bin4_R = bin_evoked['evk_bin4'].copy().pick(Rch)
        evk_bin4_L = bin_evoked['evk_bin4'].copy().pick(Lch)
        evk_bin5_R = bin_evoked['evk_bin5'].copy().pick(Rch)
        evk_bin5_L = bin_evoked['evk_bin5'].copy().pick(Lch)
        evk_bin6_R = bin_evoked['evk_bin6'].copy().pick(Rch)
        evk_bin6_L = bin_evoked['evk_bin6'].copy().pick(Lch)

        # Define functions to create the new bin operations
        def bin_operator(data1, data2):
            return 0.5 * data1 + 0.5 * data2
        
        # Create the new bins
        nbin1 = bin_operator(evk_bin1_R.data, evk_bin2_L.data)
        nbin2 = bin_operator(evk_bin1_L.data, evk_bin2_R.data)
        nbin3 = bin_operator(evk_bin3_R.data, evk_bin4_L.data)
        nbin4 = bin_operator(evk_bin3_L.data, evk_bin4_R.data)
        nbin5 = bin_operator(evk_bin5_R.data, evk_bin6_L.data)
        nbin6 = bin_operator(evk_bin5_L.data, evk_bin6_R.data)
        
        # Useful to plot the data
        time = bin_evoked['evk_bin1'].times * 1000  # Convert to milliseconds

        # Define the channel indices for (P7, P9, and) PO7
        #P7_idx = bin_evoked['evk_bin1'].info['ch_names'].index('P7')
        #P9_idx = bin_evoked['evk_bin1'].info['ch_names'].index('P9')
        PO7_idx = bin_evoked['evk_bin1'].info['ch_names'].index('PO7')

        # Extract the data for (P7, P9, and) PO7 electrodes
        PO7_data_nbin1 = nbin1[PO7_idx]
        PO7_data_nbin2 = nbin2[PO7_idx]
        PO7_data_nbin3 = nbin3[PO7_idx]
        PO7_data_nbin4 = nbin4[PO7_idx]
        PO7_data_nbin5 = nbin5[PO7_idx]
        PO7_data_nbin6 = nbin6[PO7_idx]
        
        return PO7_data_nbin1, PO7_data_nbin2, PO7_data_nbin3, PO7_data_nbin4, PO7_data_nbin5, PO7_data_nbin6, time
    
    
    if subject_id == 'GA':
        
        subject_list = get_non_excluded_subjects_list(excluded_subjects_list, input_dir, exclude_subjects=exclude_subjects)
        
        if population == 'control':

            # keep only the control subjects (i.e. subject IDs < 50)
            subject_list = [subject for subject in subject_list if int(subject[-2:]) < 50]
            print(f'====================== subjects list, control population : {subject_list}')
            

        elif population == 'stroke':

            # keep only the stroke subjects (i.e. subject IDs > 50)
            subject_list = [subject for subject in subject_list if int(subject[-2:]) > 50]
            print(f'====================== subjects list, stroke population : {subject_list}')
            
        # initialize lists to store the data for each subject
        PO7_data_nbin1_list = []
        PO7_data_nbin2_list = []
        PO7_data_nbin3_list = []
        PO7_data_nbin4_list = []
        PO7_data_nbin5_list = []
        PO7_data_nbin6_list = []
        
        for subject in subject_list:
            
            sub_id = subject[-2:]
            b1, b2, b3, b4, b5, b6, time = get_evoked_data(sub_id, input_dir)
            
            PO7_data_nbin1_list.append(b1)
            PO7_data_nbin2_list.append(b2)
            PO7_data_nbin3_list.append(b3)
            PO7_data_nbin4_list.append(b4)
            PO7_data_nbin5_list.append(b5)
            PO7_data_nbin6_list.append(b6)
            print(f'====================== data collected for {subject}')
        
        # transform the lists in np.arrays
        PO7_data_nbin1_array = np.array(PO7_data_nbin1_list)
        PO7_data_nbin2_array = np.array(PO7_data_nbin2_list)
        PO7_data_nbin3_array = np.array(PO7_data_nbin3_list)
        PO7_data_nbin4_array = np.array(PO7_data_nbin4_list)
        PO7_data_nbin5_array = np.array(PO7_data_nbin5_list)
        PO7_data_nbin6_array = np.array(PO7_data_nbin6_list)
        
        # compute the mean of each array (should be length evk.info['sfreq']*0.6 (200ms basline, 400ms after 0))
        PO7_data_nbin1_mean = PO7_data_nbin1_array.mean(axis=0)
        PO7_data_nbin2_mean = PO7_data_nbin2_array.mean(axis=0)
        PO7_data_nbin3_mean = PO7_data_nbin3_array.mean(axis=0)
        PO7_data_nbin4_mean = PO7_data_nbin4_array.mean(axis=0)
        PO7_data_nbin5_mean = PO7_data_nbin5_array.mean(axis=0)
        PO7_data_nbin6_mean = PO7_data_nbin6_array.mean(axis=0)
        
        return PO7_data_nbin1_mean, PO7_data_nbin2_mean, PO7_data_nbin3_mean, PO7_data_nbin4_mean, PO7_data_nbin5_mean, PO7_data_nbin6_mean, time
            
    else:
        
        b1, b2, b3, b4, b5, b6, time = get_evoked_data(subject_id, input_dir)
        
        return b1, b2, b3, b4, b5, b6, time

def plot_n2pc(subject_id, input_dir, output_dir, exclude_subjects=False, excluded_subjects_list=[], population=None):
    ''' This function plots the N2pc ERP for a given subject, or a population if you specify subject_id = 'GA'.

    Parameters
    ----------
    subject_id : str
        The subject ID to plot. Can be 'GA' to plot the grand average.
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.
    exclude_subjects : bool
        Whether to exclude subjects from the grand average.
    excluded_subjects_list : list
        List of subjects to be excluded from the grand average.
    population : str
        Population (control or stroke).

    Returns
    -------
    None
    '''
    # define a function to create the plots
    def create_erp_plot(subject_id, contra, ipsi, time, color, condition, title, output_dir):

        plt.figure(figsize=(10, 6))
        plt.plot(time, contra, color=color, label=f'{condition} (Contralateral)')
        plt.plot(time, ipsi, color=color, linestyle='dashed', label=f'{condition} (Ipsilateral)')
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linewidth=1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (uV)')
        plt.title(f'Signal from Electrodes PO7 - {condition} Condition')
        plt.legend()
        plt.grid()
        if subject_id == 'GA':
            plt.savefig(os.path.join(output_dir, 'all_subj','n2pc-plots', population, f'{title}.png'))
        else:
            plt.savefig(os.path.join(output_dir, f'sub-{subject_id}','n2pc-plots',f'sub-{subject_id}-PO7_{condition}.png'))
        plt.show(block=False)
        plt.close()
        
        
    if subject_id == 'GA':
        
        d1, d2, d3, d4, d5, d6, time = get_bins_data(subject_id, input_dir, exclude_subjects=exclude_subjects, excluded_subjects_list=excluded_subjects_list, population=population)
        # Create output directory if it doesn't exist
        if os.path.exists(os.path.join(output_dir, 'all_subj','n2pc-plots', population)) == False:
            os.makedirs(os.path.join(output_dir, 'all_subj','n2pc-plots', population))
        
        if population == 'control':
            str_excluded_subjects_list = [str(sub) for sub in excluded_subjects_list if sub < 50]
            excluded_subjects_string = '_'.join(str_excluded_subjects_list)
        elif population == 'stroke':
            str_excluded_subjects_list = [str(sub) for sub in excluded_subjects_list if sub > 50]
            excluded_subjects_string = '_'.join(str_excluded_subjects_list)
        
    else:
        
        d1, d2, d3, d4, d5, d6, time = get_bins_data(subject_id, input_dir)
        # Create output directory if it doesn't exist
        if os.path.exists(os.path.join(output_dir, f'sub-{subject_id}','n2pc-plots')) == False:
            os.makedirs(os.path.join(output_dir, f'sub-{subject_id}','n2pc-plots'))
        
        excluded_subjects_string = ''
        population = ''
            
        
    condition1 = 'Dis_Mid'
    condition2 = 'No_Dis'
    condition3 = 'Dis_Contra'
    
    if exclude_subjects == True:
        title1 = f'{population}-excluded_subjects-{excluded_subjects_string}-PO7-{condition1}'
    else:
        title1 = f'{population}-PO7-{condition1}'
    if exclude_subjects == True:
        title2 = f'{population}-excluded_subjects-{excluded_subjects_string}-PO7-{condition2}'
    else:
        title2 = f'{population}-PO7-{condition2}'
    if exclude_subjects == True:
        title3 = f'{population}-excluded_subjects-{excluded_subjects_string}-PO7-{condition3}'
    else:
        title3 = f'{population}-PO7-{condition3}'

    create_erp_plot(subject_id, d1, d2, time, 'blue', condition1, title1, output_dir)
    create_erp_plot(subject_id, d3, d4, time,'green', condition2, title2, output_dir)
    create_erp_plot(subject_id, d5, d6, time, 'red', condition3, title3, output_dir)


def get_n2pc_values(subject_id, input_dir, output_dir, exclude_subjects=False, excluded_subjects_list=[], population=None):
    '''
    This function extracts the N2pc values for a given subject and saves them in a csv file. 
    Can be used to extract the values for all subjects (subject_id = 'GA').
    
    Parameters
    ----------
    subject_id : str
        The subject ID to plot.
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.
    exclude_subjects : bool
        Whether to exclude subjects from the grand average.
    excluded_subjects_list : list
        List of subjects to be excluded from the grand average.
    population : str
        Population (control or stroke).

    Returns
    -------
    df : pandas.DataFrame
        A dataframe containing the N2pc values for each condition and side.
    '''
    if subject_id == 'GA':
        b1, b2, b3, b4, b5, b6, time = get_bins_data(subject_id, input_dir, exclude_subjects=exclude_subjects, excluded_subjects_list=excluded_subjects_list, population=population)

    else:
        b1, b2, b3, b4, b5, b6, time = get_bins_data(subject_id, input_dir)

    bin_list = [b1, b2, b3, b4, b5, b6]

    # Define 50ms windows lists
    slices_150_200 = []
    slices_200_250 = []
    slices_250_300 = []
    slices_300_350 = []
    slices_350_400 = []
    # Define 100ms and 200ms windows lists
    slices_200_300 = []
    slices_300_400 = []
    slices_200_400 = []

    # 51 refers to the number of sample points in 100ms (sfreq = 512)
    t_150 = 51*3.5
    t_150 = math.ceil(t_150)
    t_200 = 51*4
    t_250 = 51*4.5
    t_250 = math.ceil(t_250)
    t_300 = 51*5
    t_350 = 51*5.5
    t_350 = math.ceil(t_350)
    t_400 = 51*6

    for bin_ in bin_list:

        # Slice the data into 50ms windows
        window_150_200 = bin_[t_150:t_200].mean()
        window_200_250 = bin_[t_200:t_250].mean()
        window_250_300 = bin_[t_250:t_300].mean()
        window_300_350 = bin_[t_300:t_350].mean()
        window_350_400 = bin_[t_350:t_400].mean()
       # Slice the data into 100ms and 200ms windows
        window_200_300 = bin_[t_200:t_300].mean()
        window_300_400 = bin_[t_300:t_400].mean()
        window_200_400 = bin_[t_200:t_400].mean()
        
        # Append the slices to the respective lists
        slices_150_200.append(window_150_200)
        slices_200_250.append(window_200_250)
        slices_250_300.append(window_250_300)
        slices_300_350.append(window_300_350)
        slices_350_400.append(window_350_400)
        slices_200_300.append(window_200_300)
        slices_300_400.append(window_300_400)
        slices_200_400.append(window_200_400)
        
    # Create the dataframe and store the values
    bin_names = ['Dis_mid (Contra)',
                'Dis_mid (Ipsi)',
                'No_dis (Contra)',
                'No_dis (Ipsi)',
                'Dis_contra (Contra)',
                'Dis_contra (Ipsi)']
    
    df = pd.DataFrame({'ID':subject_id,'condition and side':bin_names,
                       '150-200ms':slices_150_200,
                        '200-250ms':slices_200_250,
                        '250-300ms':slices_250_300,
                        '300-350ms':slices_300_350,
                        '350-400ms':slices_350_400,
                        '200-300ms':slices_200_300,
                        '300-400ms':slices_300_400,
                        'total 200-400ms':slices_200_400})
    
    pd.options.display.float_format = '{:.5e}'.format
    
    # Save the dataframe
    if subject_id == 'GA':
        if not os.path.exists(os.path.join(output_dir, 'all_subj', 'n2pc-values', population)):
            os.makedirs(os.path.join(output_dir, 'all_subj', 'n2pc-values', population))

        if population == 'control':
            str_excluded_subjects_list = [str(sub) for sub in excluded_subjects_list if sub < 50]
            excluded_subjects_string = '_'.join(str_excluded_subjects_list)
        elif population == 'stroke':
            str_excluded_subjects_list = [str(sub) for sub in excluded_subjects_list if sub > 50]
            excluded_subjects_string = '_'.join(str_excluded_subjects_list)
            
        title = f'{population}-excluded_subjects-{excluded_subjects_string}'

        df.to_csv(os.path.join(output_dir, 'all_subj', 'n2pc-values', population, f'{title}-n2pc_values.csv'))
    else:
        if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'n2pc-values')):
            os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'n2pc-values'))
        df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'n2pc-values', f'sub-{subject_id}-n2pc_values.csv'))
    
    return df



########## Getting values per epoch ##########

def get_df_n2pc_values_epoch(subject_id, input_dir, output_dir):
    ''' Compute the difference bewteen the ipsi and contra channels (PO7-PO8) for each epoch and save the values in a dataframe

    Parameters
    ----------
    subject_id : str
        The subject ID to plot.
    input_dir : str
        The path to the directory containing the input data.

    Returns
    -------
    df : pandas.DataFrame
        A dataframe containing the N2pc values for each condition and side.
    '''

    # load data
    file = os.path.join(input_dir, f'sub-{subject_id}/cleaned_epochs/sub-{subject_id}-cleaned_epochs-N2pc.fif')
    epochs = mne.read_epochs(file)

    # crop epochs to relevent time window
    epochs.crop(tmin=0, tmax=0.4)
    
    # get the reeject log (preprocessing step) for the subject
    reject_log = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'preprocessing', 'step-06-final-reject_log', f'sub-{subject_id}-final-reject_log-N2pc.npz'))
    # Define the epochs status (rejected or not)
    epochs_status = reject_log['bad_epochs']

    # initialize the df
    df = pd.DataFrame(columns=['ID','epoch_index', 'epoch_dropped', 'condition', 'target_side', '150-200ms', '200-250ms', '250-300ms', '300-350ms', '350-400ms', '200-300ms', '300-400ms', 'total 200-400ms', 'index_reset'])
    
    # create row for each epoch
    df['epoch_index'] = range(1,len(epochs_status)+1)
    df['epoch_dropped'] = epochs_status
    df['ID'] = subject_id

    # add a column that store the reset index of the epochs
    index_val = 0
    index_list = []
    n_valid = []
    # iterate through the 'epoch_dropped' column to create the reset index column
    for row_number in range(len(df)):
        if df.iloc[row_number, 2] == False:
            index_list.append(index_val)
            n_valid.append(index_val)
            index_val += 1
        else:
            index_list.append(np.nan)

    # add the index column to the DataFrame
    df['index_reset'] = index_list

    # iterate through the rows of the DataFrame to fill the columns
    for row_number in range(len(df)):
        
        # check if the epoch is dropped
        if df.iloc[row_number, 2] == True:
            print(f'========= epoch {row_number+1} was dropped',)
        else:
            print(f'========= epoch {row_number+1} was keeped')

            # compute the data to fill the dataframe

            # get the epoch index (after epochs rejection)
            epoch_idx = int(df['index_reset'].loc[row_number])
            
            # get the data from the channels of interest
            PO7 = epochs[epoch_idx].get_data(picks=['PO7'])
            PO8 = epochs[epoch_idx].get_data(picks=['PO8'])
            PO7 = PO7.reshape(206)
            PO8 = PO8.reshape(206)
            
            # find where is ispsilateral and contralateral to the target
            epoch_id = epochs.events[epoch_idx][2]
            if epoch_id in [1, 3, 5, 7]:
                target_side = 'left'
                ipsi = PO7
                contra = PO8
            elif epoch_id in [2, 4, 6, 8]:
                target_side = 'right'
                ipsi = PO8
                contra = PO7

            # get the difference between the channels
            diff = ipsi - contra
            
            if epoch_id in [1, 2, 5, 6]:
                cond = 'Dis_mid'
            elif epoch_id in [3, 4]:
                cond = 'No_dis'
            elif epoch_id in [7, 8]:
                cond = 'Dis_contra'

            # create the time windows based on sfreq
            sfreq = epochs.info['sfreq'] 
            t_150 = sfreq * 0.15
            t_150 = math.ceil(t_150)
            t_200 = sfreq * 0.2
            t_200 = math.ceil(t_200)
            t_250 = sfreq * 0.25
            t_250 = math.ceil(t_250)
            t_300 = sfreq * 0.3
            t_300 = math.ceil(t_300)
            t_350 = sfreq * 0.35
            t_350 = math.ceil(t_350)
            t_400 = sfreq * 0.4
            t_400 = math.ceil(t_400)

            # slice the data into 50ms and 100ms windows
            diff_150_200 = diff[t_150:t_200].mean()
            diff_200_250 = diff[t_200:t_250].mean()
            diff_250_300 = diff[t_250:t_300].mean()
            diff_300_350 = diff[t_300:t_350].mean()
            diff_350_400 = diff[t_350:t_400].mean()
            diff_200_300 = diff[t_200:t_300].mean()
            diff_300_400 = diff[t_300:t_400].mean()
            diff_200_400 = diff[t_200:t_400].mean()

            # fill the dataframe with everything we just computed 
            df.iloc[row_number, 3] = cond
            df.iloc[row_number, 4] = target_side
            df.iloc[row_number, 5] = diff_150_200
            df.iloc[row_number, 6] = diff_200_250
            df.iloc[row_number, 7] = diff_250_300
            df.iloc[row_number, 8] = diff_300_350
            df.iloc[row_number, 9] = diff_350_400
            df.iloc[row_number, 10] = diff_200_300
            df.iloc[row_number, 11] = diff_300_400
            df.iloc[row_number, 12] = diff_200_400
    
    print(f'========== df created for subject {subject_id}')

    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'n2pc-values')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'n2pc-values'))
    df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'n2pc-values', f'sub-{subject_id}-n2pc_values_per_epoch.csv'))

    print(f'========== df saved for subject {subject_id}')



##### LEGACY CODE #####

def get_population_files(input_dir, task, population):

    files = glob.glob(os.path.join(input_dir, 'all_subj', f'{population}_allsubj', '*'))
    files = [file for file in files if task in file]

    return files

def get_population_epochs(files, exclude_subjects=False, excluded_subjects_list=[]):

    def get_last_string(file):
        return file.split('/')[-1].split('-')[-1].split('.')[0]

    if exclude_subjects == True:
        excluded_subjects_list = [str(sub) for sub in excluded_subjects_list]
        excluded_subjects_string = '_'.join(excluded_subjects_list)
        specific_file = [file for file in files if get_last_string(file) == excluded_subjects_string]
    
    if exclude_subjects == False:
        files = files.sort()
        specific_file = files[0]

    epochs = mne.read_epochs(specific_file)
    
    return epochs, excluded_subjects_string

