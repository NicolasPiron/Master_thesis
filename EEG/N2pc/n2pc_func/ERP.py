import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import math

def to_evoked(subject_id, input_dir):
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
    file = os.path.join(input_dir, f'sub-{subject_id}/N2pc/cleaned_epochs/sub-{subject_id}-cleaned_epochs-N2pc.fif')
    epochs = mne.read_epochs(file)

    # crop the epochs to the relevant time window
    tmin = -0.2
    tmax = 0.8
    epochs.crop(tmin=tmin, tmax=tmax)
        
    # define the bins
    bins = {'bin1' : ['dis_top/target_l','dis_bot/target_l'],
            'bin2' : ['dis_top/target_r','dis_bot/target_r'],
            'bin3' : ['no_dis/target_l'],
            'bin4' : ['no_dis/target_r'],
            'bin5' : ['dis_right/target_l'],
            'bin6' : ['dis_left/target_r'],
            'bin7' : ['dis_mid/target_l', 'dis_bot/target_l', 'no_dis/target_l', 'dis_right/target_l'],
            'bin8' : ['dis_mid/target_r', 'dis_bot/target_r', 'no_dis/target_r', 'dis_left/target_r']}

    # create evoked for each bin
    evoked_list = [epochs[bin].average() for bin in bins.values()]

    # create evoked for all the conditions
    evoked_all = epochs.average()
    
    # rename the distractor mid conditions to simplify
    evoked_1 = evoked_list[0]
    evoked_2 = evoked_list[1]
    evoked_7 = evoked_list[6]
    evoked_8 = evoked_list[7]
    evoked_1.comment = 'dis_mid/target_l'
    evoked_2.comment = 'dis_mid/target_r'
    evoked_7.comment = 'target_l'
    evoked_8.comment = 'target_r'
    
    # replace the '/' that causes problems when saving
    for evoked in evoked_list:
        evoked.comment = evoked.comment.replace('/', '_')
    

    # save the evoked objects in subject directory
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc'))
    for evoked in evoked_list:
        print(evoked.comment)
        evoked.save(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc', f'sub-{subject_id}-{evoked.comment}-ave.fif'), overwrite=True)
    evoked_all.save(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc', f'sub-{subject_id}-all-ave.fif'), overwrite=True)


def to_evoked_population(input_dir, output_dir, subject_list, population):

    # create lists of evoked objects for each condition
    dis_mid_target_l_list = []
    dis_mid_target_r_list = []
    no_dis_target_l_list = []
    no_dis_target_r_list = []
    dis_right_target_l_list = []
    dis_left_target_r_list = []
    target_l_list = []
    target_r_list = []

    for subject in subject_list:
        file = os.path.join(input_dir, f'sub-{subject}/N2pc/cleaned_epochs/sub-{subject}-cleaned_epochs-N2pc.fif')
        epochs = mne.read_epochs(file)
        # crop the epochs to the relevant time window
        tmin = -0.2
        tmax = 0.8
        epochs.crop(tmin=tmin, tmax=tmax)
        
        # get the evoked objects for each condition
        dis_mid_target_l = epochs['dis_top/target_l','dis_bot/target_l'].average()
        dis_mid_target_r = epochs['dis_top/target_r','dis_bot/target_r'].average()
        no_dis_target_l = epochs['no_dis/target_l'].average()
        no_dis_target_r = epochs['no_dis/target_r'].average()
        dis_right_target_l = epochs['dis_right/target_l'].average()
        dis_left_target_r = epochs['dis_left/target_r'].average()
        target_l = epochs['dis_mid/target_l', 'dis_bot/target_l', 'no_dis/target_l', 'dis_right/target_l'].average()
        target_r = epochs['dis_mid/target_r', 'dis_bot/target_r', 'no_dis/target_r', 'dis_left/target_r'].average()

        # append the evoked objects to the lists
        dis_mid_target_l_list.append(dis_mid_target_l)
        dis_mid_target_r_list.append(dis_mid_target_r)
        no_dis_target_l_list.append(no_dis_target_l)
        no_dis_target_r_list.append(no_dis_target_r)
        dis_right_target_l_list.append(dis_right_target_l)
        dis_left_target_r_list.append(dis_left_target_r)
        target_l_list.append(target_l)
        target_r_list.append(target_r)

    # combine the evoked objects
    dis_mid_target_l_combined = mne.combine_evoked(dis_mid_target_l_list, weights='equal')
    dis_mid_target_r_combined = mne.combine_evoked(dis_mid_target_r_list, weights='equal')
    no_dis_target_l_combined = mne.combine_evoked(no_dis_target_l_list, weights='equal')
    no_dis_target_r_combined = mne.combine_evoked(no_dis_target_r_list, weights='equal')
    dis_right_target_l_combined = mne.combine_evoked(dis_right_target_l_list, weights='equal')
    dis_left_target_r_combined = mne.combine_evoked(dis_left_target_r_list, weights='equal')
    target_l_combined = mne.combine_evoked(target_l_list, weights='equal')
    target_r_combined = mne.combine_evoked(target_r_list, weights='equal')

    dis_mid_target_l_combined.comment = 'dis_mid_target_l'
    dis_mid_target_r_combined.comment = 'dis_mid_target_r'
    no_dis_target_l_combined.comment = 'no_dis_target_l'
    no_dis_target_r_combined.comment = 'no_dis_target_r'
    dis_right_target_l_combined.comment = 'dis_right_target_l'
    dis_left_target_r_combined.comment = 'dis_left_target_r'
    target_l_combined.comment = 'target_l'
    target_r_combined.comment = 'target_r'

    # save the combined evoked objects
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', population)):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', population))

    dis_mid_target_l_combined.save(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', population, f'{population}-dis_mid_target_l-ave.fif'), overwrite=True)
    dis_mid_target_r_combined.save(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', population, f'{population}-dis_mid_target_r-ave.fif'), overwrite=True)
    no_dis_target_l_combined.save(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', population, f'{population}-no_dis_target_l-ave.fif'), overwrite=True)
    no_dis_target_r_combined.save(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', population, f'{population}-no_dis_target_r-ave.fif'), overwrite=True)
    dis_right_target_l_combined.save(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', population, f'{population}-dis_right_target_l-ave.fif'), overwrite=True)
    dis_left_target_r_combined.save(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', population, f'{population}-dis_left_target_r-ave.fif'), overwrite=True)
    target_l_combined.save(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', population, f'{population}-target_l-ave.fif'), overwrite=True)
    target_r_combined.save(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', population, f'{population}-target_r-ave.fif'), overwrite=True)

    return None

def combine_evoked_all(input_dir, subject_list):
    ''' Takes a list of subjects and concatenates the evoked objects for each subject.

    Parameters
    ----------
    subject_list : list
        List of subjects to be concatenated. [01, 02, 03, ...] format.
    input_dir : str
        The path to the directory containing the input data.
    
    Returns
    -------
    evk_all : mne.Evoked
        The combined evoked object.
    
    '''
    
    # Get the evoked objects for each subject
    evk_list = []
    for subject_id in subject_list:
        try:
            evk = mne.read_evokeds(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc', f'sub-{subject_id}-all-ave.fif'))
            evk_list.append(evk[0])
        except:
            print(f'====================== No evoked file found for subject {subject_id}')
            continue
    # Concate the evoked objects
    evk_all = mne.combine_evoked(evk_list, weights='equal')
    print('====================== evoked objects combined for all conditions')

    return evk_all

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
    evoked_path = os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', f'evoked-N2pc')
    evoked_files = glob.glob(os.path.join(evoked_path, f'sub-{subject_id}-*.fif'))
    # Load the evoked files
    evoked_list = [mne.read_evokeds(evoked_file)[0] for evoked_file in evoked_files]
    bin_dict = {'bin1' : 'dis_mid_target_l',
            'bin2' : 'dis_mid_target_r',
            'bin3' : 'no_dis_target_l',
            'bin4' : 'no_dis_target_r',
            'bin5' : 'dis_right_target_l',
            'bin6' : 'dis_left_target_r',
            'bin7' : 'target_l',
            'bin8' : 'target_r'}

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

def get_swapped_evoked_patient(subject_id, input_dir):
    ''' This function swaps the right and left conditions for the patients that have a lesion on the right side.
        It is used to have the same lateralization for all the patients.
        The only diff with get_evoked is that it inverts the names of the conditions and the comments of the evoked objects.
    '''

    subject_id = str(subject_id)
    evoked_path = os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', f'evoked-N2pc')
    evoked_files = glob.glob(os.path.join(evoked_path, f'sub-{subject_id}-*.fif'))
    # Load the evoked files
    evoked_list = [mne.read_evokeds(evoked_file)[0] for evoked_file in evoked_files]
    bin_dict = {'bin1' : 'dis_mid_target_l',
            'bin2' : 'dis_mid_target_r',
            'bin3' : 'no_dis_target_l',
            'bin4' : 'no_dis_target_r',
            'bin5' : 'dis_right_target_l',
            'bin6' : 'dis_left_target_r',
            'bin7' : 'target_l',
            'bin8' : 'target_r'}

    # mapping of the conditions to swap
    swapped_dict = {'dis_mid_target_l' : 'dis_mid_target_r',
            'dis_mid_target_r' : 'dis_mid_target_l',
            'no_dis_target_l' : 'no_dis_target_r',
            'no_dis_target_r' : 'no_dis_target_l',
            'dis_right_target_l' : 'dis_left_target_r',
            'dis_left_target_r' : 'dis_right_target_l',
            'target_l' : 'target_r',
            'target_r' : 'target_l'}

    # Assign the evoked object that corresponds to the bin
    swapped_bin_evoked = {}

    # start by inverting the side of the comment
    for original_comment, new_comment in swapped_dict.items():
        for evoked in evoked_list:
            if evoked.comment == original_comment:
                evoked.comment = new_comment
                break 
                
    # then do as in get_evoked()            
    for bin_name, comment in bin_dict.items():
        for evoked in evoked_list:
            if evoked.comment == comment:
                swapped_bin_evoked[bin_name] = evoked
                break 

    # Rename the keys of the dict
    prefix = 'evk_'
    # Create a new dictionary with modified keys
    swapped_bin_evoked = {prefix + key: value for key, value in swapped_bin_evoked.items()}

    return swapped_bin_evoked

def combine_evoked_single_subj(subject_id, input_dir, output_dir):  
        # load the evoked files
        bin_evoked = get_evoked(subject_id, input_dir)

        print(bin_evoked)

        # create to list of evoked objects for each condition
        dis_mid = [bin_evoked[bin_].crop(tmin=0, tmax=0.8) for bin_ in ['evk_bin1', 'evk_bin2']]
        no_dis = [bin_evoked[bin_].crop(tmin=0, tmax=0.8) for bin_ in ['evk_bin3', 'evk_bin4']]
        dis_contra = [bin_evoked[bin_].crop(tmin=0, tmax=0.8) for bin_ in ['evk_bin5', 'evk_bin6']]

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
            if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', f'evoked-N2pc', 'combined')):
                os.makedirs(os.path.join(output_dir ,f'sub-{subject_id}', 'N2pc', f'evoked-N2pc', 'combined'))
            combined_pair.save(os.path.join(output_dir, f'sub-{subject_id}','N2pc', f'evoked-N2pc', 'combined', f'sub-{subject_id}-{pair_names[i]}-ave.fif'), overwrite=True)

def combine_swapped_evoked_patient(subject_id, input_dir, output_dir):

    bin_evoked = get_swapped_evoked_patient(subject_id, input_dir)

    print(bin_evoked)

    # create to list of evoked objects for each condition
    dis_mid = [bin_evoked[bin_].crop(tmin=0, tmax=0.8) for bin_ in ['evk_bin1', 'evk_bin2']]
    no_dis = [bin_evoked[bin_].crop(tmin=0, tmax=0.8) for bin_ in ['evk_bin3', 'evk_bin4']]
    dis_contra = [bin_evoked[bin_].crop(tmin=0, tmax=0.8) for bin_ in ['evk_bin5', 'evk_bin6']]

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
        if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', f'evoked-N2pc', 'swapped')):
            os.makedirs(os.path.join(output_dir ,f'sub-{subject_id}', 'N2pc', f'evoked-N2pc', 'swapped'))
        combined_pair.save(os.path.join(output_dir, f'sub-{subject_id}','N2pc', f'evoked-N2pc', 'swapped', f'sub-{subject_id}-{pair_names[i]}-ave.fif'), overwrite=True)

def combine_evoked_population(input_dir, output_dir, subject_list, population):
    '''
    Parameters
    ----------
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.
    subject_list : list of str
        List of subjects to be concatenated. ['01', '02', '03', ...] format.
    population : str
        Population, can be 'thal_control', 'young_control', 'old_control' or 'pulvinar'.

    Returns
    -------
    None
    '''

    # create lists of evoked objects for each condition 
    dis_mid_list = []
    no_dis_list = []
    dis_contra_list = []

    for subject_id in subject_list:

        # check is the combined evoked files already exist
        if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc', 'combined', f'sub-{subject_id}-dis_mid-ave.fif')):
            
            combine_evoked_single_subj(subject_id, input_dir, output_dir)
            print(f'====================== evoked files combined for {subject_id}')
        else:
            print(f'====================== evoked files were already combined for {subject_id}')
        
        # loop over the subjects and append the evoked objects to the lists
        evoked_files = glob.glob(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', f'evoked-N2pc', 'combined', f'sub-{subject_id}*ave.fif'))
        print(evoked_files)
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
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', 'combined', population)):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', 'combined', population))

    dis_mid_combined.save(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', 'combined', population, f'{population}-dis_mid-ave.fif'), overwrite=True)
    no_dis_combined.save(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', 'combined', population, f'{population}-no_dis-ave.fif'), overwrite=True)
    dis_contra_combined.save(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', 'combined', population, f'{population}-dis_contra-ave.fif'), overwrite=True)
    
def combine_topo_diff_single_subj(subject_id, input_dir, output_dir):
    '''
    computes and saves the contra minus ipsi scalp for a given subject, for each condition.

    Parameters
    ----------
    subject_id : str
        The subject ID to plot. e.g. '01'
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.

    Returns
    -------
    None
    '''

    evk_dict = get_evoked(subject_id, input_dir)
    print(evk_dict)

    # find right and left channels
    ch_names = list(evk_dict.values())[0].info['ch_names']
    LCh = []
    RCh = []
    LPO = []
    RPO = []
    for i, ch in enumerate(ch_names):
        if str(ch[-1]) == 'z':
            pass
        elif int(ch[-1]) % 2 == 0:
            RCh.append(i)
            if ch in ['PO8', 'PO4', 'O2']:
                RPO.append(i)
        elif int(ch[-1]) %2 != 2:
            if ch in ['PO7', 'PO3', 'O1']:
                LPO.append(i)
            LCh.append(i) 

    diff_evk = {}
    for bin_, evk in evk_dict.items():
        data = evk.copy().get_data()
        if 'target_l' in evk.comment:
            data[RPO] = data[LPO]
        elif 'target_r' in evk.comment:
            data[LPO] = data[RPO]
        swapped = mne.EvokedArray(data, evk.info, tmin=evk.times[0], nave=evk.nave)
        diff_evk[bin_] = mne.combine_evoked([evk, swapped], weights=[1, -1])
        diff_evk[bin_].comment = evk.comment + '_diff'
        diff_evk[bin_].nave = evk.nave

    print(diff_evk)

    # group the evoked objects by condition
    dis_mid = [diff_evk[bin_].crop(tmin=0, tmax=0.8) for bin_ in ['evk_bin1', 'evk_bin2']]
    no_dis = [diff_evk[bin_].crop(tmin=0, tmax=0.8) for bin_ in ['evk_bin3', 'evk_bin4']]
    dis_contra = [diff_evk[bin_].crop(tmin=0, tmax=0.8) for bin_ in ['evk_bin5', 'evk_bin6']]

    # swap the right and left channels for the right target evoked objects
    pairs = [dis_mid, no_dis, dis_contra]
    pair_names = ['dis_mid_diff', 'no_dis_diff', 'dis_contra_diff']
    for i, pair in enumerate(pairs):
        # the right target evoked object will be laterally swapped so it is like the target is on the left
        to_swap = pair[1]
        data = to_swap.copy().get_data()
        left = data[LCh]
        right = data[RCh]
        data[RCh] = left
        data[LCh] = right
        swapped = mne.EvokedArray(data, to_swap.info, tmin=to_swap.times[0], nave=to_swap.nave)
        combined_pair = mne.combine_evoked([pair[0], swapped], weights='equal')
        combined_pair.comment = pair_names[i]
        diff_evk[pair_names[i]] = combined_pair
    
    print(diff_evk)

    # save the combined evoked objects
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc', 'diff')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc', 'diff'))
    for bin_, evk in diff_evk.items():
        evk.save(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc', 'diff', f'sub-{subject_id}-{evk.comment}-ave.fif'), overwrite=True)

def combine_topo_diff_population(input_dir, output_dir, subject_list, population):
    '''
    Parameters
    ----------
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.
    subject_list : list of str
        List of subjects to be concatenated. ['01', '02', '03', ...] format.
    population : str
        Population, can be 'thal_control', 'young_control', 'old_control' or 'pulvinar'.

    Returns
    -------
    None
    '''

    # create lists of evoked objects for each condition
    dis_mid_target_l_list = []
    dis_mid_target_r_list = []
    no_dis_target_l_list = []
    no_dis_target_r_list = []
    dis_right_target_l_list = []
    dis_left_target_r_list = []
    dis_mid_list = []
    no_dis_list = []
    dis_contra_list = []
    target_l_list = []
    target_r_list = []

    list_dict = {'dis_mid_target_l_diff' : dis_mid_target_l_list,
                    'dis_mid_target_r_diff' : dis_mid_target_r_list,
                    'no_dis_target_l_diff' : no_dis_target_l_list,
                    'no_dis_target_r_diff' : no_dis_target_r_list,
                    'dis_right_target_l_diff' : dis_right_target_l_list,
                    'dis_left_target_r_diff' : dis_left_target_r_list,
                    'dis_mid_diff' : dis_mid_list,
                    'no_dis_diff' : no_dis_list,
                    'dis_contra_diff' : dis_contra_list,
                    'target_l_diff' : target_l_list,
                    'target_r_diff' : target_r_list}

    for subject_id in subject_list:

        # check is the combined evoked files already exist
        if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc', 'diff', f'sub-{subject_id}-dis_mid_diff-ave.fif')):
            
            combine_topo_diff_single_subj(subject_id, input_dir, output_dir)
            print(f'====================== evoked files combined for {subject_id}')
        else:
            print(f'====================== evoked files were already combined for {subject_id}')
        
        # loop over the subjects and append the evoked objects to the lists
        evoked_files = glob.glob(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', f'evoked-N2pc', 'diff', '*.fif'))
        evoked_list = [mne.read_evokeds(evoked_file)[0] for evoked_file in evoked_files]
        evoked_dict = {}
        for evoked in evoked_list:
            evoked_dict[evoked.comment] = evoked
            print(evoked.comment)

        for key, list_ in list_dict.items():
            list_.append(evoked_dict[key])

    # combine the evoked objects
    combined_dic = {}
    for key, list_ in list_dict.items():
        combined_dic[key] = mne.combine_evoked(list_, weights='equal')
        combined_dic[key].comment = key

    # save the combined evoked objects
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', 'diff', population)):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', 'diff', population))
    for key, evk in combined_dic.items():
        evk.save(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', 'diff', population, f'{population}-{key}-ave.fif'), overwrite=True)


def load_combined_evoked(evoked_list):
    '''
    Used to load the combined evoked objects into a dictionary. -> plot topo maps
    '''
    evoked_dict = {}
    for evoked in evoked_list:
        evoked_dict[evoked.comment] = evoked
    return evoked_dict
    

def plot_erp_topo_single_subj(subject_id, input_dir, output_dir):
    '''
    Parameters
    ----------
    subject_id : str
        The subject ID to plot. 
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.

    Returns
    -------
    None
    '''

    subject_id = str(subject_id)

    print(f'====================== plotting for {subject_id}')

    # load data and store it in a dictionary
    evoked_diff_files = glob.glob(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc', 'diff', f'sub-{subject_id}*.fif'))
    evoked_files = glob.glob(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc', f'sub-{subject_id}*.fif'))
    if len(evoked_files) == 0:
        print('====================== no file found - evoked files (not combined)')
    if len(evoked_diff_files) == 0:
        print('====================== no file found - evoked files (diff)')
    all_evoked_files = evoked_diff_files + evoked_files
    evoked_list = [mne.read_evokeds(evoked_file)[0] for evoked_file in all_evoked_files]
    evoked_dict = load_combined_evoked(evoked_list)

    # plot the topomaps
    for bin_, evoked in evoked_dict.items():
        # reset the bad channels
        evoked.info['bads'] = []
        topo = evoked.plot_topomap(times=[0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26], show=False)
        if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'n2pc-topo', 'diff')):
            os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'n2pc-topo', 'diff'))
        bin_name = bin_.replace('/', '_')
        if len(bin_name) > 50:
            bin_name = 'all' # to avoid errors when saving the file
        if 'diff' in bin_name:
            topo.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'n2pc-topo', 'diff', f'sub-{subject_id}-topo-{bin_name}.png'))
        else:
            topo.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'n2pc-topo', f'sub-{subject_id}-topo-{bin_name}.png'))
        print(f'====================== topo plot saved for {subject_id} - {bin_name}')
        plt.close()

    # plot the topomaps but means of time windows
    for bin_, evoked in evoked_dict.items():
        # reset the bad channels
        evoked.info['bads'] = []
        topo = evoked.plot_topomap(times=[0.175, 0.225, 0.275, 0.325, 0.375], average=0.05, show=False)
        if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'n2pc-topo', 'diff', 'mean')):
            os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'n2pc-topo', 'diff', 'mean'))
        if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'n2pc-topo', 'mean')):
            os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'n2pc-topo', 'mean'))
        bin_name = bin_.replace('/', '_')
        if len(bin_name) > 50:
            bin_name = 'all'
        if 'diff' in bin_name:
            topo.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'n2pc-topo', 'diff', 'mean', f'sub-{subject_id}-topo-{bin_name}.png'))
        else:
            topo.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'n2pc-topo', 'mean', f'sub-{subject_id}-topo-{bin_name}.png'))
        print(f'====================== mean topo plot saved for {subject_id} - {bin_name}')

    

def plot_erp_topo_population(input_dir, output_dir, population):
    '''
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

    # load data and store it in a dictionary
    evoked_files = glob.glob(os.path.join(input_dir, 'all_subj', 'N2pc', 'evoked-N2pc', population, f'{population}*.fif'))
    evoked_list = [mne.read_evokeds(evoked_file)[0] for evoked_file in evoked_files]
    evoked_dict = load_combined_evoked(evoked_list)
    # plot the topomaps
    for bin_, evoked in evoked_dict.items():
        # reset the bad channels
        evoked.info['bads'] = []
        topo = evoked.plot_topomap(times=[0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26], vlim=(-5, 5), show=False)
        if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-plots', population, 'n2pc-topo')):
            os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-plots', population, 'n2pc-topo'))
        bin_name = bin_.replace('/', '_')
        topo.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-plots', population, 'n2pc-topo', f'{population}-topo-{bin_name}.png'))
        print(f'====================== topo plot saved for {population} - {bin_}')

    # do the same but at different scale to visualize the ipsi and contra components
    evoked_diff_files = glob.glob(os.path.join(input_dir, 'all_subj', 'N2pc', 'evoked-N2pc', 'diff', population, f'{population}*.fif'))
    evoked_diff_list = [mne.read_evokeds(evoked_file)[0] for evoked_file in evoked_diff_files]
    evoked_diff_dict = load_combined_evoked(evoked_diff_list)
    # plot the topomaps
    for bin_, evoked in evoked_diff_dict.items():
        # reset the bad channels
        evoked.info['bads'] = []
        if 'target' in bin_:
            vlim = (-2, 2)
        else:
            vlim = (-1, 1)
        topo1 = evoked.plot_topomap(times=[0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26], vlim=vlim, show=False)
        topo2 = evoked.plot_topomap(times=[0.175, 0.225, 0.275, 0.325, 0.375], average=0.05,vlim=vlim, res=256, show=False)
        if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-plots', population, 'n2pc-topo', 'diff', 'mean')):
            os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-plots', population, 'n2pc-topo', 'diff', 'mean'))
        bin_name = bin_.replace('/', '_')
        topo1.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-plots', population, 'n2pc-topo', 'diff', f'{population}-topo-{bin_name}.png'))
        topo2.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-plots', population, 'n2pc-topo', 'diff', 'mean', f'{population}-topo-{bin_name}.png'))
        print(f'====================== topo plot saved for {population} - {bin_}')


def plot_n2pc_both_sides_single_subj(subject_id, input_dir, output_dir):
    '''
    function to visualize the n2pc for the 6 condition without doing the mean across both sides of the head.

    Parameters
    ----------
    subject_id : str
        The subject ID to plot.
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.
    '''

    evoked_files = glob.glob(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc', f'sub-{subject_id}*.fif')) 
    evoked_list = [mne.read_evokeds(evoked_file)[0] for evoked_file in evoked_files]
    evoked_dict = load_combined_evoked(evoked_list)

    time = list(evoked_dict.values())[0].times*1000

    for key, evk in evoked_dict.items():

        if len(key) > 50:
            continue
        if 'target_l' in evk.comment:
            contra = evk.copy().get_data(picks=['PO8'])
            contra = contra.reshape(contra.shape[1],1)
            ipsi = evk.copy().get_data(picks=['PO7'])
            ipsi = ipsi.reshape(ipsi.shape[1],1)
        elif 'target_r' in evk.comment:
            contra = evk.copy().get_data(picks=['PO7'])
            contra = contra.reshape(contra.shape[1],1)
            ipsi = evk.copy().get_data(picks=['PO8'])
            ipsi = ipsi.reshape(ipsi.shape[1],1)
        fig, ax = plt.subplots()
        ax.plot(time, contra, label='contra')
        ax.plot(time, ipsi, label='ipsi')
        ax.plot(time, contra-ipsi, label='diff')
        ax.set_title(key)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (uV)')
        ax.legend()
        if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'n2pc-waveform', 'n2pc-6-conds')):
            os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'n2pc-waveform', 'n2pc-6-conds'))
        bin_name = key.replace('/', '_')
        fig.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'n2pc-waveform', 'n2pc-6-conds', f'sub-{subject_id}-n2pc-{bin_name}.png'))
        print(f'====================== waveform plot saved for {subject_id} - {bin_name}')

def plot_n2pc_both_sides_population(input_dir, output_dir, population):
    '''
    cf plot_n2pc_both_sides_single_subj
    '''

    evoked_files = glob.glob(os.path.join(input_dir, 'all_subj', 'N2pc', 'evoked-N2pc', population, f'{population}*.fif'))
    evoked_list = [mne.read_evokeds(evoked_file)[0] for evoked_file in evoked_files]
    evoked_dict = load_combined_evoked(evoked_list)

    time = list(evoked_dict.values())[0].times*1000

    for key, evk in evoked_dict.items():

        if len(key) > 50:
            continue
        if 'target_l' in evk.comment:
            contra = evk.copy().get_data(picks=['PO8'])
            contra = contra.reshape(contra.shape[1],1)
            ipsi = evk.copy().get_data(picks=['PO7'])
            ipsi = ipsi.reshape(ipsi.shape[1],1)
        elif 'target_r' in evk.comment:
            contra = evk.copy().get_data(picks=['PO7'])
            contra = contra.reshape(contra.shape[1],1)
            ipsi = evk.copy().get_data(picks=['PO8'])
            ipsi = ipsi.reshape(ipsi.shape[1],1)
        fig, ax = plt.subplots()
        ax.plot(time, contra, label='contra')
        ax.plot(time, ipsi, label='ipsi')
        ax.plot(time, contra-ipsi, label='diff')
        ax.set_title(key)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (uV)')
        ax.legend()
        if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-plots', population, 'n2pc-waveform', 'n2pc-6-conds')):
            os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-plots', population, 'n2pc-waveform', 'n2pc-6-conds'))
        bin_name = key.replace('/', '_')
        fig.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-plots', population, 'n2pc-waveform', 'n2pc-6-conds', f'{population}-n2pc-{bin_name}.png'))
        print(f'====================== waveform plot saved for {population} - {bin_name}')


def get_evoked_data_single_subj(subject_id, input_dir):    
    ''' This function extracts the N2pc ERP values for a given subject.

    Parameters
    ----------
    subject_id : str
        The subject ID OR 'GA' to get the grand average.
    input_dir : str
        The path to the directory containing the input data.
    
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

def get_evoked_data_population(input_dir, subject_list):
    '''
    Gets the N2pc ERP data for a population of subjects.

    Parameters
    ----------
    input_dir : str
        The path to the directory containing the input data.
    subject_list : list of str
        List of subjects to be concatenated. ['01', '02', '03', ...] format.

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
        
    # initialize lists to store the data for each subject
    PO7_data_nbin1_list = []
    PO7_data_nbin2_list = []
    PO7_data_nbin3_list = []
    PO7_data_nbin4_list = []
    PO7_data_nbin5_list = []
    PO7_data_nbin6_list = []
    
    for subject_id in subject_list:
        
        b1, b2, b3, b4, b5, b6, time = get_evoked_data_single_subj(subject_id, input_dir)
        
        PO7_data_nbin1_list.append(b1)
        PO7_data_nbin2_list.append(b2)
        PO7_data_nbin3_list.append(b3)
        PO7_data_nbin4_list.append(b4)
        PO7_data_nbin5_list.append(b5)
        PO7_data_nbin6_list.append(b6)
        print(f'====================== data collected for {subject_id}')
    
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

def get_diff(contra, ipsi):
    diff_waveform = contra - ipsi
    return diff_waveform


def plot_n2pc_all_cond_single_subj(subject_id, input_dir, output_dir):
    ''' This function plots the N2pc ERP based on the basic evoked object for a given subject, or all of them. 

    Parameters
    ----------
    subject_id : str
        The subject ID OR 'GA' to get the grand average.
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.

    Returns
    -------
    None
    '''

    subject_id = str(subject_id)

    # get the epochs, average them and plot the ERP
    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))
    evoked = epochs.average()

    # plot the ERP for all trials (all conditions) 
    PO7 = evoked.get_data(picks=['PO7'])
    PO8 = evoked.get_data(picks=['PO8'])
    PO7 = PO7.reshape(PO7.shape[1], 1)
    PO8 = PO8.reshape(PO8.shape[1], 1)
    time = evoked.times * 1000

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, PO8, label='PO8')
    ax.plot(time, PO7, label='PO7')
    ax.plot(time, PO8 - PO7, color='cyan', label='PO8 - PO7')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (uV)')
    ax.set_title('Signal from Electrodes PO7 and PO8 - All Conditions')
    ax.legend()
    ax.grid()

    # save the plot
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'n2pc-waveform')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'n2pc-waveform'))
    fig.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'n2pc-waveform', f'sub-{subject_id}-all-conditions.png'))

    # save the evoked object
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc', 'all_cond')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc', 'all_cond'))
    evoked.save(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc', 'all_cond', f'sub-{subject_id}-all-conditions-ave.fif'), overwrite=True)

    return None

def plot_n2pc_all_cond_population(input_dir, output_dir, subject_list, population):
    ''' This function plots the N2pc ERP based on the basic evoked object for a given subject, or all of them. 
        If subject_id = 'GA', the function will plot the grand average. There is only one condition = all the trials are considered.

    Parameters
    ----------
    subject_id : str
        The subject ID OR 'GA' to get the grand average.
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.
    exclude_subjects : bool
        Whether to exclude subjects from the grand average.
    excluded_subjects_list : list
        List of subjects to be excluded from the grand average.
    population : str
        Population (old_control, young_control or stroke).
    
    Returns
    -------
    None
    '''

    evoked = combine_evoked_all(input_dir, subject_list)

    # plot the ERP for all trials (all conditions) # there is no need to differentiate between contra and ipsi
    # this is why we do it in a very simple way. 
    PO7 = evoked.get_data(picks=['PO7'])
    PO8 = evoked.get_data(picks=['PO8'])
    PO7 = PO7.reshape(PO7.shape[1], 1)
    PO8 = PO8.reshape(PO8.shape[1], 1)
    time = evoked.times * 1000

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, PO8, label='PO8')
    ax.plot(time, PO7, label='PO7')
    ax.plot(time, PO8 - PO7, color='cyan', label='PO8 - PO7')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (uV)')
    ax.set_title('Signal from Electrodes PO7 and PO8 - All Conditions')
    ax.legend()
    ax.grid()

    # save the plot
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-plots', population, 'n2pc-waveform')):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-plots', population, 'n2pc-waveform'))
    fig.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-plots', population, 'n2pc-waveform', f'{population}-all-conditions.png'))
    
    # save the evoked object
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', 'all_cond', population)):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', 'all_cond', population))
    evoked.save(os.path.join(output_dir, 'all_subj', 'N2pc', 'evoked-N2pc', 'all_cond', population, f'{population}-all-conditions-ave.fif'), overwrite=True)

    return None

def plot_n2pc_single_subj(subject_id, input_dir, output_dir):
    ''' This function plots the N2pc ERP for a given subject.

    Parameters
    ----------
    subject_id : str
        The subject ID to plot. Can be 'GA' to plot the grand average.
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str

    Returns
    -------
    None
    '''
    # define a function to create the plots
    def create_erp_plot(subject_id, contra, ipsi, time, color, condition, output_dir):

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
        plt.savefig(os.path.join(output_dir, f'sub-{subject_id}','N2pc','n2pc-plots', 'n2pc-waveform', f'sub-{subject_id}-PO7_{condition}.png'))
        plt.close()

    def create_diff_plot(subject_id, contra_cond1, ipsi_cond1, contra_cond2, ipsi_cond2, contra_cond3, ipsi_cond3, time, output_dir):

        diff_cond1 = get_diff(contra_cond1, ipsi_cond1)
        diff_cond2 = get_diff(contra_cond2, ipsi_cond2)
        diff_cond3 = get_diff(contra_cond3, ipsi_cond3)

        plt.figure(figsize=(10, 6))
        plt.plot(time, diff_cond1, color='blue', label=f'Dis_Mid')
        plt.plot(time, diff_cond2, color='green', label=f'No_Dis')
        plt.plot(time, diff_cond3, color='red', label=f'Dis_Contra')
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linewidth=1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (uV)')
        plt.title(f'Signal from Electrodes PO7-PO8 - Difference of contra vs ipsi')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f'sub-{subject_id}','N2pc','n2pc-plots', 'n2pc-waveform', f'sub-{subject_id}-diff.png'))
        plt.close()
        
    d1, d2, d3, d4, d5, d6, time = get_evoked_data_single_subj(subject_id, input_dir)
    # Create output directory if it doesn't exist
    if os.path.exists(os.path.join(output_dir, f'sub-{subject_id}','N2pc','n2pc-plots', 'n2pc-waveform')) == False:
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}','N2pc','n2pc-plots', 'n2pc-waveform'))

    create_erp_plot(subject_id, d1, d2, time, 'blue', 'Dis_Mid', output_dir)
    create_erp_plot(subject_id, d3, d4, time,'green', 'No_Dis', output_dir)
    create_erp_plot(subject_id, d5, d6, time, 'red', 'Dis_Contra', output_dir)
    create_diff_plot(subject_id, d1, d2, d3, d4, d5, d6, time, output_dir)

def plot_n2pc_population(input_dir, output_dir, subject_list, population):
    ''' This function plots the N2pc ERP for a population.

    Parameters
    ----------
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
    subject_list : list
        List of subjects to be plotted. 

    Returns
    -------
    None
    '''
    # define a function to create the plots
    def create_erp_plot(contra, ipsi, time, color, condition, title, output_dir):

        plt.figure(figsize=(10, 6))
        plt.plot(time, contra, color=color, label=f'{condition} (Contralateral)')
        plt.plot(time, ipsi, color=color, linestyle='dashed', label=f'{condition} (Ipsilateral)')
        plt.ylim(-0.000006, 0.000004)
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linewidth=1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (uV)')
        plt.title(f'Signal from Electrodes PO7 - {condition} Condition')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-plots', population, 'n2pc-waveform',  f'{title}.png'))
        plt.close()

    def create_diff_plot(contra_cond1, ipsi_cond1, contra_cond2, ipsi_cond2, contra_cond3, ipsi_cond3, time, title, output_dir):

        diff_cond1 = get_diff(contra_cond1, ipsi_cond1)
        diff_cond2 = get_diff(contra_cond2, ipsi_cond2)
        diff_cond3 = get_diff(contra_cond3, ipsi_cond3)

        plt.figure(figsize=(10, 6))
        plt.plot(time, diff_cond1, color='blue', label=f'Dis_Mid')
        plt.plot(time, diff_cond2, color='green', label=f'No_Dis')
        plt.plot(time, diff_cond3, color='red', label=f'Dis_Contra')
        plt.ylim(-0.0000018, 0.0000018)
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linewidth=1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (uV)')
        plt.title(f'Signal from Electrodes PO7-PO8 - Difference of contra vs ipsi')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-plots', population, 'n2pc-waveform' , f'diff-{title}.png'))
        plt.close()
 
    d1, d2, d3, d4, d5, d6, time = get_evoked_data_population(input_dir, subject_list)
    # Create output directory if it doesn't exist
    if os.path.exists(os.path.join(output_dir, 'all_subj','N2pc','n2pc-plots', population, 'n2pc-waveform')) == False:
        os.makedirs(os.path.join(output_dir, 'all_subj','N2pc','n2pc-plots', population, 'n2pc-waveform'))

    create_erp_plot(d1, d2, time, 'blue', 'Dis_Mid', f'{population}-PO7-Dis_Mid', output_dir)
    create_erp_plot(d3, d4, time,'green', 'No_Dis', f'{population}-PO7-No_Dis', output_dir)
    create_erp_plot(d5, d6, time, 'red', 'Dis_Contra', f'{population}-PO7-Dis_Contra', output_dir)
    create_diff_plot( d1, d2, d3, d4, d5, d6, time, f'{population}', output_dir)

def get_n2pc_values(bin_list):

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
        
    return slices_150_200, slices_200_250, slices_250_300, slices_300_350, slices_350_400, slices_200_300, slices_300_400, slices_200_400

def get_n2pc_values_single_subj(subject_id, input_dir, output_dir):
    '''
    This function extracts the N2pc values for a given subject and saves them in a csv file. 

    Parameters
    ----------
    subject_id : str
        The subject ID to plot.
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.

    -------
    df : pandas.DataFrame
        A dataframe containing the N2pc values for each condition and side.
    '''

    b1, b2, b3, b4, b5, b6, time = get_evoked_data_single_subj(subject_id, input_dir)
    bin_list = [b1, b2, b3, b4, b5, b6]
    slices_150_200, slices_200_250, slices_250_300, slices_300_350, slices_350_400, slices_200_300, slices_300_400, slices_200_400 = get_n2pc_values(bin_list)

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
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc','n2pc-values')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc','n2pc-values'))
    df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc','n2pc-values', f'sub-{subject_id}-n2pc_values.csv'))
    
    return df


def get_n2pc_values_population(input_dir, output_dir, subject_list, population):
    '''
    This function extracts the average N2pc values for a given subject list and saves them in a csv file. 
    
    Parameters
    ----------
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.
    subject_list : list
        List of subjects. [01, 02, 03, ...]
    population : str
        Population (thal_control, old_controls, young_controls or pulvinar).

    Returns
    -------
    df : pandas.DataFrame
        A dataframe containing the N2pc values for each condition and side.
    '''

    b1, b2, b3, b4, b5, b6, time = get_evoked_data_population(input_dir, subject_list)
    bin_list = [b1, b2, b3, b4, b5, b6]

    slices_150_200, slices_200_250, slices_250_300, slices_300_350, slices_350_400, slices_200_300, slices_300_400, slices_200_400 = get_n2pc_values(bin_list)
        
    # Create the dataframe and store the values
    bin_names = ['Dis_mid (Contra)',
                'Dis_mid (Ipsi)',
                'No_dis (Contra)',
                'No_dis (Ipsi)',
                'Dis_contra (Contra)',
                'Dis_contra (Ipsi)']
    
    df = pd.DataFrame({'population':population,'condition and side':bin_names,
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
    if not os.path.exists(os.path.join(output_dir, 'all_subj','N2pc', 'n2pc-values', population)):
        os.makedirs(os.path.join(output_dir, 'all_subj','N2pc', 'n2pc-values', population))
    df.to_csv(os.path.join(output_dir, 'all_subj', 'N2pc','n2pc-values', population, f'{population}-n2pc_values.csv'))

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
    file = os.path.join(input_dir, f'sub-{subject_id}/N2pc/cleaned_epochs/sub-{subject_id}-cleaned_epochs-N2pc.fif')
    epochs = mne.read_epochs(file)

    # crop epochs to relevent time window
    epochs.crop(tmin=0, tmax=0.8)
    
    # get the reeject log (preprocessing step) for the subject
    reject_log = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'preprocessing', 'step-06-final-reject_log', f'sub-{subject_id}-final-reject_log-N2pc.npz'))
    # Define the epochs status (rejected or not)
    epochs_status = reject_log['bad_epochs']

    # initialize the df
    df = pd.DataFrame(columns=['ID','epoch_index', 'epoch_dropped', 'index_reset', 'saccade', 'condition', 'target_side', '150-200ms', '200-250ms',
                                '250-300ms', '260-310ms', '270-320ms', '280-330ms', '290-340ms', '300-350ms', '350-400ms',
                                '400-450ms', '450-500ms', '500-550ms', '550-600ms', '200-300ms', '300-400ms', '400-500ms',
                                '500-600ms', '200-400ms', '300-500ms', '400-600ms', 'total 200-600ms'])
    
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

    # Load the csv file contaning the indices of epochs with saccades
    saccades = pd.read_csv(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'rejected-epochs-list', f'sub-{subject_id}-heog-artifact.csv'))
    # Create a list of the indices of the epochs with saccades
    saccades_list = list(saccades['index'])
    # Add a column that specifies if the epoch contains a saccade. FALSE if no saccade, TRUE if saccade. 
    df['saccade'] = df['index_reset'].isin(saccades_list)

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
            PO7 = PO7.reshape(410)
            PO8 = PO8.reshape(410)
            
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
            diff = contra - ipsi
            
            if epoch_id in [1, 2, 5, 6]:
                cond = 'Dis_mid'
            elif epoch_id in [3, 4]:
                cond = 'No_dis'
            elif epoch_id in [7, 8]:
                cond = 'Dis_contra'

            # create the time points based on sfreq
            sfreq = epochs.info['sfreq'] 
            t_150 = sfreq * 0.15
            t_150 = math.ceil(t_150)
            t_200 = sfreq * 0.2
            t_200 = math.ceil(t_200)
            t_250 = sfreq * 0.25
            t_250 = math.ceil(t_250)
            t_260 = sfreq * 0.26
            t_260 = math.ceil(t_260)
            t_270 = sfreq * 0.27
            t_270 = math.ceil(t_270)
            t_280 = sfreq * 0.28
            t_280 = math.ceil(t_280)
            t_290 = sfreq * 0.29
            t_290 = math.ceil(t_290)
            t_300 = sfreq * 0.3
            t_300 = math.ceil(t_300)
            t_310 = sfreq * 0.31
            t_310 = math.ceil(t_310)
            t_320 = sfreq * 0.32
            t_320 = math.ceil(t_320)
            t_330 = sfreq * 0.33
            t_330 = math.ceil(t_330)
            t_340 = sfreq * 0.34
            t_340 = math.ceil(t_340)
            t_350 = sfreq * 0.35
            t_350 = math.ceil(t_350)
            t_400 = sfreq * 0.4
            t_400 = math.ceil(t_400)
            t_450 = sfreq * 0.45
            t_450 = math.ceil(t_450)
            t_500 = sfreq * 0.5
            t_500 = math.ceil(t_500)
            t_550 = sfreq * 0.55
            t_550 = math.ceil(t_550)
            t_600 = sfreq * 0.6
            t_600 = math.ceil(t_600)

            # slice the data into 50ms and 100ms windows
            diff_150_200 = diff[t_150:t_200].mean()
            diff_200_250 = diff[t_200:t_250].mean()
            diff_250_300 = diff[t_250:t_300].mean()
            diff_260_310 = diff[t_260:t_310].mean()
            diff_270_320 = diff[t_270:t_320].mean()
            diff_280_330 = diff[t_280:t_330].mean()
            diff_290_340 = diff[t_290:t_340].mean()
            diff_300_350 = diff[t_300:t_350].mean()
            diff_350_400 = diff[t_350:t_400].mean()
            diff_400_450 = diff[t_400:t_450].mean()
            diff_450_500 = diff[t_450:t_500].mean()
            diff_500_550 = diff[t_500:t_550].mean()
            diff_550_600 = diff[t_550:t_600].mean()
            diff_200_300 = diff[t_200:t_300].mean()
            diff_300_400 = diff[t_300:t_400].mean()
            diff_400_500 = diff[t_400:t_500].mean()
            diff_500_600 = diff[t_500:t_600].mean()
            diff_200_400 = diff[t_200:t_400].mean()
            diff_300_500 = diff[t_300:t_500].mean()
            diff_400_600 = diff[t_400:t_600].mean()
            diff_200_600 = diff[t_200:t_600].mean()

            # fill the dataframe with everything we just computed 
            df.iloc[row_number, 5] = cond
            df.iloc[row_number, 6] = target_side
            df.iloc[row_number, 7] = diff_150_200
            df.iloc[row_number, 8] = diff_200_250
            df.iloc[row_number, 9] = diff_250_300
            df.iloc[row_number, 10] = diff_260_310
            df.iloc[row_number, 11] = diff_270_320
            df.iloc[row_number, 12] = diff_280_330
            df.iloc[row_number, 13] = diff_290_340
            df.iloc[row_number, 14] = diff_300_350
            df.iloc[row_number, 15] = diff_350_400
            df.iloc[row_number, 16] = diff_400_450
            df.iloc[row_number, 17] = diff_450_500
            df.iloc[row_number, 18] = diff_500_550
            df.iloc[row_number, 19] = diff_550_600
            df.iloc[row_number, 20] = diff_200_300
            df.iloc[row_number, 21] = diff_300_400
            df.iloc[row_number, 22] = diff_400_500
            df.iloc[row_number, 23] = diff_500_600
            df.iloc[row_number, 24] = diff_200_400
            df.iloc[row_number, 25] = diff_300_500
            df.iloc[row_number, 26] = diff_400_600
            df.iloc[row_number, 27] = diff_200_600
    
    print(f'========== df created for subject {subject_id}')

    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-values')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-values'))
    df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-values', f'sub-{subject_id}-n2pc_values_per_epoch.csv'))

    print(f'========== df saved for subject {subject_id}')

##### Find peak latency #####

def get_peak_latency_single_subj(subject_id, input_dir, output_dir):

    cond_list = ['dis_contra', 'dis_mid', 'no_dis', 'dis_left_target_r', 'dis_right_target_l',
                  'no_dis_target_l', 'no_dis_target_r', 'dis_mid_target_l', 'dis_mid_target_r']
    evoked_dict = {}
    for cond in cond_list:
        if 'target' in cond:
            evoked_dict[cond] = mne.read_evokeds(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc', f'sub-{subject_id}-{cond}-ave.fif'))
        else:
            evoked_dict[cond] = mne.read_evokeds(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc', 'combined', f'sub-{subject_id}-{cond}-ave.fif'))

    data_dict = {}
    info = mne.create_info(ch_names=['PO7'], sfreq=512, ch_types='eeg')
    for cond in evoked_dict.keys():
        if 'target_l' in cond:
            contra = 'PO8'
            ipsi = 'PO7'
        elif 'target_r' in cond:
            contra = 'PO7'
            ipsi = 'PO8'
        else:
            contra = 'PO8'
            ipsi = 'PO7'

        evoked = evoked_dict[cond][0].copy().crop(tmin=0)
        diff = evoked.copy().pick(contra).get_data() - evoked.copy().pick(ipsi).get_data()
        data_dict[cond] = mne.EvokedArray(diff, info, tmin=0)

    # Create a df to store the peak latencies
    df = pd.DataFrame(columns=['ID', 'condition', 'peak_latency', 'peak_amplitude', 'mean_amp_around_peak'])

    df['ID'] = np.zeros(len(data_dict))

    # define the time window
    tmin=0.18
    tmax=0.4

    # function that can find the minimum positive if there is no maximum negative in the evoked object
    def get_min_pos(evk, tmin, tmax):
        data = evk.copy().crop(tmin=tmin, tmax=tmax).get_data()
        lat = data.argmin() * (1/evk.info['sfreq']) # convert the index to time
        return lat, data.min()

    # loop through the evoked objects to get the peak latencies and amplitudes and store them in the df
    for i, evk in enumerate(data_dict.values()):
        try:
            ch, lat, amp = evk.get_peak(tmin=tmin, tmax=tmax, mode="neg", return_amplitude=True)
        # if there is no maximum negative, we look for the minimum positive
        except ValueError:
            lat, amp = get_min_pos(evk, tmin, tmax)

        upper_bound = lat + 0.025
        lower_bound = lat - 0.025
        mean_amp = evk.copy().pick('PO7').crop(tmin=lower_bound, tmax=upper_bound).get_data().mean()
        df.iloc[i, 0] = str(subject_id)
        df.iloc[i, 1] = list(data_dict.keys())[i]
        df.iloc[i, 2] = lat
        df.iloc[i, 3] = amp
        df.iloc[i, 4] = mean_amp

    # save the df
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'peak-latency')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'peak-latency'))
    df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'peak-latency', f'sub-{subject_id}-peak-latency.csv'))

def all_subjects_peak_latencies(input_dir, output_dir):
    ''' Concatenate the peak latencies of all subjects (9 conditions per subject) in a single df
    '''
    df_list = []
    dirs = os.listdir(input_dir)
    dirs.remove('all_subj')
    for directory in dirs:
        try:
            df = pd.read_csv(os.path.join(input_dir, directory, 'N2pc', 'peak-latency', f'{directory}-peak-latency.csv'), index_col=0)
            df_list.append(df)
            print(f'========= peak latency df for subject {directory} added to the list')
        except:
            print(f'========= no peak latency df for subject {directory}')
            continue

    df = pd.concat(df_list, axis=0)
    print('========= all subjects peak latencies df concatenated')
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'peak-latency')):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'peak-latency'))
    df.to_csv(os.path.join(output_dir, 'all_subj', 'N2pc', 'peak-latency', 'all_subjects_peak_latencies.csv'))

    return None

def mean_latency_per_subject(input_dir, output_dir):
    df = pd.read_csv(os.path.join(input_dir, 'all_subj', 'N2pc', 'peak-latency', 'all_subjects_peak_latencies.csv'))

    subjects = df['ID'].unique()
    mean_dict = {}

    for subject in subjects:
        condition1 = df['ID'] == subject
        condition2 = df['condition'].isin(['dis_mid', 'dis_contra', 'no_dis'])
        mean_dict[subject] = df[condition1 & condition2]['peak_latency'].mean()

    mean_df = pd.DataFrame(mean_dict.items(), columns=['ID', 'mean_peak_latency'])
    print('========= mean latency df created')
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'peak-latency')):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'peak-latency'))
    mean_df.to_csv(os.path.join(output_dir, 'all_subj', 'N2pc', 'peak-latency', 'mean_peak_latencies.csv'))

def amplitude_around_peak_by_epoch_single_subj(subject_id, input_dir, output_dir):

    file = os.path.join(input_dir, f'sub-{subject_id}/N2pc/cleaned_epochs/sub-{subject_id}-cleaned_epochs-N2pc.fif')
    epochs = mne.read_epochs(file)

    # crop epochs to relevent time window
    epochs.crop(tmin=0, tmax=0.8)
    
    # get the reeject log (preprocessing step) for the subject
    reject_log = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'preprocessing', 'step-06-final-reject_log', f'sub-{subject_id}-final-reject_log-N2pc.npz'))
    # Define the epochs status (rejected or not)
    epochs_status = reject_log['bad_epochs']

    df = pd.DataFrame(columns=['ID','epoch_index', 'epoch_dropped', 'index_reset', 'saccade', 'condition', 'target_side', 'latency', 'diff_amp', 'PO7_amp', 'PO8_amp'])

    peak_latency_df = pd.read_csv(os.path.join(input_dir, 'all_subj', 'N2pc', 'peak-latency', 'mean_peak_latencies.csv'))
    peak_latency = peak_latency_df[peak_latency_df['ID'] == int(subject_id)]['mean_peak_latency'].values[0]

    # create row for each epoch
    df['epoch_index'] = range(1,len(epochs_status)+1)
    df['epoch_dropped'] = epochs_status
    df['ID'] = subject_id
    df['latency'] = peak_latency

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

    # Load the csv file contaning the indices of epochs with saccades
    saccades = pd.read_csv(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'rejected-epochs-list', f'sub-{subject_id}-heog-artifact.csv'))
    # Create a list of the indices of the epochs with saccades
    saccades_list = list(saccades['index'])
    # Add a column that specifies if the epoch contains a saccade. FALSE if no saccade, TRUE if saccade. 
    df['saccade'] = df['index_reset'].isin(saccades_list)

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
            PO7 = PO7.reshape(410)
            PO8 = PO8.reshape(410)
            
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
            diff = contra - ipsi
            
            if epoch_id in [1, 2, 5, 6]:
                cond = 'Dis_mid'
            elif epoch_id in [3, 4]:
                cond = 'No_dis'
            elif epoch_id in [7, 8]:
                cond = 'Dis_contra'

            # get the time window based on the peak latency of the subject
            sfreq = epochs.info['sfreq']
            tmin = peak_latency - 0.020
            tmin = math.ceil(tmin * sfreq)
            tmax = peak_latency + 0.020
            tmax = math.ceil(tmax * sfreq)

            # get the amplitude around the peak
            amp = diff[tmin:tmax].mean()
            amp_PO7 = PO7[tmin:tmax].mean()
            amp_PO8 = PO8[tmin:tmax].mean()

            # fill the dataframe with everything we just computed
            df.iloc[row_number, 5] = cond
            df.iloc[row_number, 6] = target_side
            df.iloc[row_number, 8] = amp
            df.iloc[row_number, 9] = amp_PO7
            df.iloc[row_number, 10] = amp_PO8

    print(f'========== df created for subject {subject_id}')

    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-values')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-values'))
    df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-values', f'sub-{subject_id}-amplitude-around-peak.csv'))

def amplitude_around_peak_by_epoch_all_subj(input_dir, output_dir):
    ''' Takes the output files of amplitude_around_peak_by_epoch_single_subj and concat them together
    '''
    df_list = []
    dirs = os.listdir(input_dir)
    dirs.remove('all_subj')
    dirs = sorted(dirs)
    for directory in dirs:
        if not os.path.exists(os.path.join(input_dir, directory, 'N2pc', 'n2pc-values', f'{directory}-amplitude-around-peak.csv')):
            try:
                amplitude_around_peak_by_epoch_single_subj(directory[-2:], input_dir, output_dir)
            except:
                print(f'========= no amplitude around peak df for subject {directory[-2:]}')
                continue
        try:
            df = pd.read_csv(os.path.join(input_dir, directory, 'N2pc', 'n2pc-values', f'{directory}-amplitude-around-peak.csv'), index_col=0)
            df_list.append(df)
            print(f'========= amplitude around peak df for subject {directory} added to the list')
        except:
            print(f'========= no df found for subject {directory}')
    
    df = pd.concat(df_list, axis=0)
    print('========= all subjects amplitude around peak df concatenated')
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-values', 'n2pc-values-around-peak')):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-values', 'n2pc-values-around-peak'))
    df.to_csv(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-values', 'n2pc-values-around-peak', 'all_subjects_amplitude_around_peak.csv'))

    return None

def get_amp_and_power_df(input_dir, output_dir):
    ''' Create a df containing the amplitude and the alpha power of the N2pc for each subject. 

    Parameters
    ----------
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.

    Returns
    -------
    None
    '''
    amp_df = pd.read_csv(os.path.join(input_dir, 'all_subj', 'N2pc', 'n2pc-values', 'n2pc-values-around-peak', 'all_subjects_amplitude_around_peak.csv'))
    power_df = pd.read_csv(os.path.join(input_dir, 'all_subj', 'N2pc', 'alpha-power-allsubj', 'alpha-power-per-epoch-allsubj_v2.csv'))

    # take the values in the 'alpha-PO7' and 'alpha-PO8' columns for each epoch and add them to the amp_df
    power_df_subset = power_df[['ID', 'epoch_index', 'alpha-PO7', 'alpha-PO8']]
    merged_df = pd.merge(amp_df, power_df_subset, on=['epoch_index', 'ID'])

    merged_df.to_csv(os.path.join(output_dir, 'all_subj', 'N2pc', 'n2pc-values', 'n2pc-values-around-peak', 'all_subjects_amp_power.csv'))

    return None


def get_peak_latency_grand_average(input_dir, output_dir, population):

    # Get the evoked data
    dis_contra = mne.read_evokeds(os.path.join(input_dir, 'all_subj', 'N2pc', 'evoked-N2pc', 'combined', population, f'{population}-dis_contra-ave.fif'))
    dis_mid = mne.read_evokeds(os.path.join(input_dir, 'all_subj', 'N2pc', 'evoked-N2pc', 'combined', population, f'{population}-dis_mid-ave.fif'))
    no_dis = mne.read_evokeds(os.path.join(input_dir, 'all_subj', 'N2pc', 'evoked-N2pc', 'combined', population, f'{population}-no_dis-ave.fif'))

    PO7_dis_contra = dis_contra[0].copy().pick('PO7').get_data()
    PO8_dis_contra = dis_contra[0].copy().pick('PO8').get_data()
    PO7_dis_mid = dis_mid[0].copy().pick('PO7').get_data()
    PO8_dis_mid = dis_mid[0].copy().pick('PO8').get_data()
    PO7_no_dis = no_dis[0].copy().pick('PO7').get_data()
    PO8_no_dis = no_dis[0].copy().pick('PO8').get_data()

    # create the diff PO7-PO8
    diff_dis_contra = PO8_dis_contra - PO7_dis_contra
    diff_dis_mid = PO8_dis_mid - PO7_dis_mid
    diff_no_dis = PO8_no_dis - PO7_no_dis

    # back to evoked object
    info = mne.create_info(ch_names=['PO7'], sfreq=512, ch_types='eeg')
    # empty dict to store the data, easier to loop through
    peak_data = {}

    peak_data['dis_contra'] = mne.EvokedArray(diff_dis_contra, info, tmin=0)
    peak_data['dis_mid'] = mne.EvokedArray(diff_dis_mid, info, tmin=0)
    peak_data['no_dis'] = mne.EvokedArray(diff_no_dis, info, tmin=0)

    # create a df to store the peak latencies
    df = pd.DataFrame(columns=['population', 'condition', 'peak_latency', 'peak_amplitude'])

    # define the shape using the number of evoked objects
    df['population'] = np.zeros(len(peak_data))

    # define the time window
    tmin=0.18
    tmax=0.4

    # loop through the evoked objects to get the peak latencies and amplitudes and store them in the df
    for i, evk in enumerate(peak_data.values()):
        ch, lat, amp = evk.get_peak(tmin=tmin, tmax=tmax, mode="neg", return_amplitude=True)
        df.iloc[i, 0] = population
        df.iloc[i, 1] = list(peak_data.keys())[i]
        df.iloc[i, 2] = lat
        df.iloc[i, 3] = amp

    # save the df
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'peak-latency', population)):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'peak-latency', population))
    df.to_csv(os.path.join(output_dir, 'all_subj', 'N2pc', 'peak-latency', population, f'peak-latency-{population}.csv'))

    return df

def all_peak_latencies_report(input_dir, outputdir):

    # concat the peak latencies of the 4 populations
    df_list = []
    for directory in os.listdir(os.path.join(input_dir, 'all_subj', 'N2pc', 'peak-latency')):
        try:
            df = pd.read_csv(os.path.join(input_dir, 'all_subj', 'N2pc', 'peak-latency', directory, f'peak-latency-{directory}.csv'), index_col=0)
            df_list.append(df)
            print(f'========= peak latency df for population {directory} added to the list')
        except:
            print(f'========= no peak latency df for population {directory}')
            continue

    df = pd.concat(df_list, axis=0)
    print('========= all populations peak latencies df concatenated')
    if not os.path.exists(os.path.join(outputdir, 'all_subj', 'N2pc', 'peak-latency')):
        os.makedirs(os.path.join(outputdir, 'all_subj', 'N2pc', 'peak-latency'))
    df.to_csv(os.path.join(outputdir, 'all_subj', 'N2pc', 'peak-latency', 'all_populations_peak_latencies.csv'))
    print('========= all populations peak latencies df saved')

    # do the same for the single subjects
    df_list = []
    for directory in os.listdir(os.path.join(input_dir)):
        if directory.startswith('sub-'):
            try:
                df = pd.read_csv(os.path.join(input_dir, directory, 'N2pc', 'peak-latency', f'{directory}-peak-latency.csv'), index_col=0)
                df_list.append(df)
                print(f'========= peak latency df for subject {directory} added to the list')
            except:
                print(f'========= no peak latency df for subject {directory}')
                continue

    df = pd.concat(df_list, axis=0)
    print('========= all subject peak latencies df concatenated')
    df.to_csv(os.path.join(outputdir, 'all_subj', 'N2pc', 'peak-latency', 'all_subjects_peak_latencies.csv'))
    print('========= all subjects peak latencies df saved')

    return None


############################################
# P1 
############################################

def plot_P1_single_subj(subject_id, input_dir, output_dir):

    evk = mne.read_evokeds(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc', f'sub-{subject_id}-all-ave.fif'))
    evk = evk[0]
    evk.info['bads'] = []

    for ch in ['Oz', 'O1', 'O2']:

        evk_ = evk.copy().pick(ch)

        fig, ax = plt.subplots()

        evk_.plot(show=False, window_title=f'sub-{subject_id} P1', axes=ax, ylim=dict(eeg=[-10, 10]), titles=dict(eeg=ch))
        ax.grid()
        if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'p1-plots')):
            os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'p1-plots'))
        fig.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'p1-plots', f'sub-{subject_id}-P1-{ch}-ave.png'))

def plot_P1_grand_average(input_dir, output_dir, subject_list, population):

    evoked_list = list()
    for subject in subject_list:
        try:
            evk = mne.read_evokeds(os.path.join(input_dir, f'sub-{subject}', 'N2pc', 'evoked-N2pc', f'sub-{subject}-all-ave.fif'))
            evk[0].info['bads'] = []
            evoked_list.append(evk[0])
        except:
            print(f'========= no evoked found for subject {subject}')
            continue
    
    for ch in ['Oz', 'O1', 'O2', ['Oz', 'O1', 'O2']]:

        data = [evk.get_data(picks=ch) for evk in evoked_list]
        if len(ch) > 2: # if we want to plot all 3 channels (>1 is not working because 2 characters in ch name)
            ch = 'mean_occip'
            data = np.mean(data, axis=1)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        mean = mean.reshape(512)
        std = std.reshape(512)
        t = evk[0].times

        fig, ax = plt.subplots()

        #evk_.plot(show=False, window_title=f'{population} P1', axes=ax, ylim=dict(eeg=[-10, 10]), titles=dict(eeg=ch))
        ax.plot(t, mean, color='black')
        ax.fill_between(t, mean+std, mean-std, color='black', alpha=0.4)
        ax.grid()
        ax.set_title(f'{population} - P1 - {ch}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (V)')
        ax.set_ylim(-0.00001, 0.00001)
        if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'p1-plots', population)):
            os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'p1-plots', population))
        fig.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'p1-plots', population, f'{population}-P1-{ch}-ave.png'))

def get_P1_latency_single_subj(subject_id, input_dir, output_dir):

    # Load the evoked data
    evk = mne.read_evokeds(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'evoked-N2pc', f'sub-{subject_id}-all-ave.fif'))
    evk.info['bads'] = []
    evk = evk[0].copy().pick('Oz')

    tmin = 0.
    tmax = 0.2

    ch, lat, amp = evk.get_peak(tmin=tmin, tmax=tmax, mode="pos", return_amplitude=True)

    # Create a df to store the peak latencies
    df = pd.DataFrame(columns=['ID', 'peak_latency', 'peak_amplitude'])
    df['ID'] = np.zeros(1)
    df.iloc[0, 0] = subject_id
    df.iloc[0, 1] = lat
    df.iloc[0, 2] = amp

    # save the df
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'peak-latency')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'peak-latency'))
    df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'peak-latency', f'sub-{subject_id}-P1-peak-latency.csv'))

def P1_amp_around_peak_per_epoch_single_subj(subject_id, input_dir, output_dir):

    file = os.path.join(input_dir, f'sub-{subject_id}/N2pc/cleaned_epochs/sub-{subject_id}-cleaned_epochs-N2pc.fif')
    epochs = mne.read_epochs(file)
    epochs.info['bads'] = []

    # crop epochs to relevent time window
    epochs.crop(tmin=0, tmax=0.8)
    
    # get the reeject log (preprocessing step) for the subject
    reject_log = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'preprocessing', 'step-06-final-reject_log', f'sub-{subject_id}-final-reject_log-N2pc.npz'))
    # Define the epochs status (rejected or not)
    epochs_status = reject_log['bad_epochs']

    df = pd.DataFrame(columns=['ID','epoch_index', 'epoch_dropped', 'index_reset', 'saccade', 'condition', 'target_side', 'latency','amp_Oz', 'amp_O1', 'amp_O2'])

    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'peak-latency', f'sub-{subject_id}-P1-peak-latency.csv')):
        get_P1_latency_single_subj(subject_id, input_dir, output_dir)
    peak_latency_df = pd.read_csv(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'peak-latency', f'sub-{subject_id}-P1-peak-latency.csv'))
    peak_latency = peak_latency_df['peak_latency'].values[0]

    df['epoch_index'] = range(1,len(epochs_status)+1)
    df['epoch_dropped'] = epochs_status
    df['ID'] = subject_id
    df['latency'] = peak_latency

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

    # Load the csv file contaning the indices of epochs with saccades
    saccades = pd.read_csv(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'rejected-epochs-list', f'sub-{subject_id}-heog-artifact.csv'))
    # Create a list of the indices of the epochs with saccades
    saccades_list = list(saccades['index'])
    # Add a column that specifies if the epoch contains a saccade. FALSE if no saccade, TRUE if saccade. 
    df['saccade'] = df['index_reset'].isin(saccades_list)

    for row_number in range(len(df)):

                # check if the epoch is dropped
        if df.iloc[row_number, 2] == True:
            print(f'========= epoch {row_number+1} was dropped',)
        else:

            # get the epoch index (after epochs rejection)
            epoch_idx = int(df['index_reset'].loc[row_number])

            epoch_id = epochs.events[epoch_idx][2]
            if epoch_id in [1, 3, 5, 7]:
                target_side = 'left'
            elif epoch_id in [2, 4, 6, 8]:
                target_side = 'right'
            
            if epoch_id in [1, 2, 5, 6]:
                cond = 'Dis_mid'
            elif epoch_id in [3, 4]:
                cond = 'No_dis'
            elif epoch_id in [7, 8]:
                cond = 'Dis_contra'

            sfreq = epochs.info['sfreq']
            tmin = peak_latency - 0.020
            tmin = math.ceil(tmin * sfreq)
            tmax = peak_latency + 0.020
            tmax = math.ceil(tmax * sfreq)

            amp_Oz = epochs[epoch_idx].get_data(picks=['Oz']).reshape(410)[tmin:tmax].mean()
            amp_O1 = epochs[epoch_idx].get_data(picks=['O1']).reshape(410)[tmin:tmax].mean()
            amp_O2 = epochs[epoch_idx].get_data(picks=['O2']).reshape(410)[tmin:tmax].mean()

            # fill the dataframe with everything we just computed
            df.iloc[row_number, 5] = cond
            df.iloc[row_number, 6] = target_side
            df.iloc[row_number, 8] = amp_Oz
            df.iloc[row_number, 9] = amp_O1
            df.iloc[row_number, 10] = amp_O2

    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'p1-values')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'p1-values'))
    df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'p1-values', f'sub-{subject_id}-p1-amplitude-around-peak.csv'))

def P1_pipeline_single_subj(subject_id, input_dir, output_dir):

    plot_P1_single_subj(subject_id, input_dir, output_dir)
    get_P1_latency_single_subj(subject_id, input_dir, output_dir)
    P1_amp_around_peak_per_epoch_single_subj(subject_id, input_dir, output_dir)


def P1_amp_around_peak_per_epoch_all_subj(input_dir, output_dir):

    df_list = []
    dirs = os.listdir(input_dir)
    dirs.remove('all_subj')
    dirs = sorted(dirs)
    for directory in dirs:
        if not os.path.exists(os.path.join(input_dir, directory, 'N2pc', 'p1-values', f'{directory}-p1-amplitude-around-peak.csv')):
            try:
                P1_amp_around_peak_per_epoch_single_subj(directory[-2:], input_dir, output_dir)
            except:
                print(f'========= no P1 amplitude around peak df for subject {directory[-2:]}')
                continue
        try:
            df = pd.read_csv(os.path.join(input_dir, directory, 'N2pc', 'p1-values', f'{directory}-p1-amplitude-around-peak.csv'), index_col=0)
            df_list.append(df)
            print(f'========= amplitude around peak df for subject {directory} added to the list')
        except:
            print(f'========= no df found for subject {directory}')
    
    df = pd.concat(df_list, axis=0)
    print('========= all subjects amplitude around peak df concatenated')
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'p1-values', 'p1-values-around-peak')):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'p1-values', 'p1-values-around-peak'))
    df.to_csv(os.path.join(output_dir, 'all_subj', 'N2pc', 'p1-values', 'p1-values-around-peak', 'all_subjects_P1_amplitude_around_peak.csv'))


############################################
# Functions to get a long format df for seaborn

def compute_diff(subject_id, cond, input_dir, swapped=False):
    
    if swapped:
        evk_dir = 'swapped'
    else:
        evk_dir = 'combined'
    
    path = os.path.join(input_dir, f'sub-{subject_id}', 'N2pc',
                        'evoked-N2pc', evk_dir, f'sub-{subject_id}-{cond}-ave.fif' )
    evoked = mne.read_evokeds(path, verbose=False)
    evoked = evoked[0]
    po7 = evoked.get_data(picks='PO7')
    po7 = po7.reshape(po7.shape[1])
    po8 = evoked.get_data(picks='PO8')
    po8 = po8.reshape(po8.shape[1])
    diff = po7 - po8
    t = evoked.times
    
    return diff, t
    
def get_group(subject_id):
    
    grp_mapping = {'old':[1, 2, 3, 4, 6, 7, 12,  13, 16, 17, 18, 19, 20, 21, 22, 23],
                  'pulvinar':[51, 53, 59, 60],
                  'thalamus':[52, 54, 55, 56, 58]}
    
    for group, ids in grp_mapping.items():
        if int(subject_id) in ids:
            return group
    
def create_long_n2pc_df_subj(subject_id, cond, input_dir, swapped=False):
    
    diff, t = compute_diff(subject_id, cond, input_dir, swapped=swapped)
    s_id = [subject_id]*len(t)
    grp = [get_group(subject_id)]*len(t)
    cond = [cond]*len(t)
    data = {'time':t,
            'signal':diff,
            'ID':s_id,
            'group':grp,
            'condition':cond
           }
    df = pd.DataFrame(data)
    
    return df
    
def create_long_n2pc_df(subjects, input_dir):
    
    df_list = list()
    conds = ['dis_mid', 'no_dis', 'dis_contra']
    right_lesion_patients = [51, 53, 54, 58, 59]
        
    for subject_id in subjects:
        try:
            if int(subject_id) in right_lesion_patients:
                for cond in conds:
                    df = create_long_n2pc_df_subj(subject_id, cond, input_dir, swapped=True)
                    df_list.append(df)
            else:
                for cond in conds:
                    df = create_long_n2pc_df_subj(subject_id, cond, input_dir, swapped=False)
                    df_list.append(df)
        except:
            print(f'No data for subject {subject_id}')
            continue
           
    full_df = pd.concat(df_list, axis=0)
    if not os.path.exists(os.path.join(input_dir, 'all_subj', 'N2pc', 'n2pc-values', 'long_format_n2pc')):
        os.makedirs(os.path.join(input_dir, 'all_subj', 'N2pc', 'n2pc-values', 'long_format_n2pc'))
    full_df.to_csv(os.path.join(input_dir, 'all_subj', 'N2pc', 'n2pc-values', 'long_format_n2pc', 'long_format_n2pc.csv'))