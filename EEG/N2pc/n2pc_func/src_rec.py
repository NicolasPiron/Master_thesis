import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from set_paths import get_paths
from mne import read_labels_from_annot
from mne.datasets import fetch_fsaverage

def source_set_up():
    '''
    Set up the source space, the BEM model and the forward solution for the first subject. 
    This is the same for every subject because we don't have MRI scans, so other functions will 
    always use this forward solution.

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''

    input_dir, o = get_paths()
    info = mne.io.read_info(os.path.join(input_dir, 'sub-01', 'N2pc', 'cleaned_epochs','sub-01-cleaned_epochs-N2pc.fif'))
    info['bads'] = []

    # get the fsaverage directory. There is a different one for each machine becauses I wasn't able to fetch from on the server.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if 'nicolaspiron/Documents' in script_dir:
        subjects_dir = fetch_fsaverage(verbose=True)
    elif 'shared_PULSATION' in script_dir:
        subjects_dir = '/home/nicolasp/shared_PULSATION/MNE-fsaverage-data/fsaverage'
    else:
        raise Exception('Please specify the path to the fsaverage directory in the source_set_up function.')
    
    src = mne.setup_source_space(subject='', spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
    model = mne.make_bem_model(subject='', subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    fwd = mne.make_forward_solution(info, trans='fsaverage', src=src, bem=bem, eeg=True, mindist=5.0)
    if not os.path.exists(os.path.join(input_dir, 'sub-01', 'fwd')):
        os.makedirs(os.path.join(input_dir, 'sub-01', 'fwd'))
    mne.write_forward_solution(os.path.join(input_dir, 'sub-01', 'fwd', 'sub-01-fwd.fif'), fwd, overwrite=True)

def create_stc_epochs(subject_id):
    '''
    Create the source time course epochs for a given subject.

    Parameters
    ----------
    subject_id : str
        The subject id. 2 digits format, e.g. '01'.

    Returns
    -------
    None
    '''

    input_dir, o = get_paths()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if 'nicolaspiron/Documents' in script_dir:
        subjects_dir = fetch_fsaverage(verbose=True)
    elif 'shared_PULSATION' in script_dir:
        subjects_dir = '/home/nicolasp/shared_PULSATION/MNE-fsaverage-data/fsaverage'
    else:
        raise Exception('Please specify the path to the fsaverage directory in the create_stc_epochs function.')
    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))
    info = epochs.info
    info['bads'] = []

    # the same forward for every subject becauses no MRI scans
    if not os.path.exists(os.path.join(input_dir, 'sub-01', 'fwd', 'sub-01-fwd.fif')):
        source_set_up()

    # create stc epochs directory
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs'))
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', 'labels')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', 'labels'))

    # compute noise covariance matrix and inverse operator
    fwd = mne.read_forward_solution(os.path.join(input_dir, 'sub-01', 'fwd', 'sub-01-fwd.fif'))
    cov = mne.compute_covariance(epochs)
    inverse_operator = mne.minimum_norm.make_inverse_operator(info, fwd, cov)

    # save the cov and inverse operator
    mne.write_cov(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', f'{subject_id}-cov.fif'), cov)
    mne.minimum_norm.write_inverse_operator(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', f'{subject_id}-inv.fif'), inverse_operator)

    # apply inverse operator
    lambda2 = 1. / 9.
    method = 'sLORETA'
    epochs.set_eeg_reference('average', projection=True)
    epochs.apply_proj()
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, lambda2=lambda2, method=method, pick_ori=None)
    
    # extract labels from the aparc parcellation
    labels = read_labels_from_annot('', parc='aparc', subjects_dir=subjects_dir)
    labels = [label for label in labels if 'unknown' not in label.name]
    # get the label time course for each epoch -> n_epochs x n_labels x n_times
    label_ts_epochs = [mne.extract_label_time_course(stc, labels, fwd['src'], mode='pca_flip') for stc in stcs]
    del stcs
    label_ts_array = np.array(label_ts_epochs) 
    np.save(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', f'sub-{subject_id}-stc_epochs.npy'), label_ts_array)
    
    # save the events in a npy file to be able to index them later
    events = epochs.events
    np.save(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', f'sub-{subject_id}-events.npy'), events)

def split_stcs_into_cond(subject_id):
    '''
    Takes the source time course epochs (np.array) and splits them into conditions. This uses the events.npy file
    where the events are stored as a list. The indices of the events that correspond to the condition are used to
    index the epochs.

    Parameters
    ----------
    subject_id : str
        The subject id. 2 digits format, e.g. '01'.
    
    Returns
    -------
    None
    '''

    input_dir, o = get_paths()
    events = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', f'sub-{subject_id}-events.npy'))
    # get only the event ids for each epoch
    events = events[:,2].tolist()
    label_ts_array = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', f'sub-{subject_id}-stc_epochs.npy'))

    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', 'conditions')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', 'conditions'))

    conditions = {'dis_top_target_l':1,
            'dis_top_target_r':2,
            'no_dis_target_l':3,
            'no_dis_target_r':4,
            'dis_bot_target_l':5,
            'dis_bot_target_r':6,
            'dis_right_target_l':7,
            'dis_left_target_r':8,
            }
    
    cond_data = dict()
    for condition, event_id in conditions.items():
        # get the indices of the events that correspond to the condition
        indices = [i for i, x in enumerate(events) if x == event_id]
        # get the label time course for each epoch -> n_epochs x n_labels x n_times
        label_ts_epochs = label_ts_array[indices]
        cond_data[condition] = label_ts_epochs
    
    # combine the dis_top and dis_bot conditions -> dis_mid
    cond_data['dis_mid_target_l'] = np.concatenate((cond_data['dis_top_target_l'], cond_data['dis_bot_target_l']), axis=0)
    cond_data['dis_mid_target_r'] = np.concatenate((cond_data['dis_top_target_r'], cond_data['dis_bot_target_r']), axis=0)

    del cond_data['dis_top_target_l']
    del cond_data['dis_top_target_r']
    del cond_data['dis_bot_target_l']
    del cond_data['dis_bot_target_r']

    for key in cond_data.keys():
        np.save(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', 'conditions', f'sub-{subject_id}-{key}-stc_epochs.npy'), cond_data[key])
        print(f'========== {key} saved ==========')
    
def load_stcs_conditions(subject_id):
    '''
    Returns a dictionary with the source time course epochs for each condition.
    '''
    input_dir, o = get_paths()
    conditions = ['dis_mid_target_l',
            'dis_mid_target_r',
            'no_dis_target_l',
            'no_dis_target_r',
            'dis_right_target_l',
            'dis_left_target_r']
    
    cond_data = dict()
    for condition in conditions:
        cond_data[condition] = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', 'conditions', f'sub-{subject_id}-{condition}-stc_epochs.npy'))
    return cond_data

def combine_conditions(subject_id):

    combined_cond_data = dict()
    cond_data = load_stcs_conditions(subject_id)
    combined_cond_data['dis_mid'] = np.concatenate((cond_data['dis_mid_target_l'], cond_data['dis_mid_target_r']), axis=0)
    combined_cond_data['no_dis'] = np.concatenate((cond_data['no_dis_target_l'], cond_data['no_dis_target_r']), axis=0)
    combined_cond_data['dis_lat'] = np.concatenate((cond_data['dis_right_target_l'], cond_data['dis_left_target_r']), axis=0)

    return combined_cond_data

def average_stcs(subject_id):
    '''
    Average the epochs together like in MNE. n_epochs x n_labels x n_times -> n_labels x n_times.
    '''

    input_dir, o = get_paths()
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', 'evoked')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', 'evoked'))

    cond_data = load_stcs_conditions(subject_id)
    for condition, data in cond_data.items():
        cond_data[condition] = np.mean(data, axis=0)
        np.save(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', 'evoked', f'sub-{subject_id}-{condition}-stc_epochs.npy'), cond_data[condition])
        print(f'========== evoked {condition} saved ==========')
    return None

def load_evoked_stc(subject_id):
    '''
    Returns a dictionary with the evoked source time course epochs for each condition.
    '''
    input_dir, o = get_paths()
    conditions = ['dis_mid_target_l',
            'dis_mid_target_r',
            'no_dis_target_l',
            'no_dis_target_r',
            'dis_right_target_l',
            'dis_left_target_r']
    
    cond_data = dict()
    for condition in conditions:
        cond_data[condition] = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', 'evoked', f'sub-{subject_id}-{condition}-stc_epochs.npy'))
        print(f'========== evoked {condition} loaded ==========')
    return cond_data

def plot_n2pc_stc(subject_id):
    '''
    Create exploratory plots for the N2pc in source space.
    '''
    input_dir, o = get_paths()
    cond_data = load_evoked_stc(subject_id)
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', 'evoked', 'plots')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', 'evoked', 'plots'))

    # get the indices of the labels
    inferiorparietal_lh = 14
    inferiorparietal_rh = 15
    lateraloccipital_lh = 22
    lateraloccipital_rh = 23

    for cond, data in cond_data.items():
        # get the time course for the labels
        inferiorparietal_lh_ts = data[inferiorparietal_lh]
        inferiorparietal_rh_ts = data[inferiorparietal_rh]
        lateraloccipital_lh_ts = data[lateraloccipital_lh]
        lateraloccipital_rh_ts = data[lateraloccipital_rh]

        # get the time points for the mean time course
        times = np.linspace(-200, 800, 512)

        # plot the mean time course for the labels
        plt.plot(times, inferiorparietal_lh_ts, label='inferiorparietal_lh')
        plt.plot(times, inferiorparietal_rh_ts, label='inferiorparietal_rh')
        plt.plot(times, lateraloccipital_lh_ts, label='lateraloccipital_lh')
        plt.plot(times, lateraloccipital_rh_ts, label='lateraloccipital_rh')
        plt.legend()
        plt.title(f'{subject_id} {cond}')
        plt.savefig(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', 'evoked', 'plots', f'{subject_id}-{cond}.png'))
        plt.close()

def plot_n2pc_population_stc(subject_list):
    '''
    Create exploratory plots for the N2pc in source space for a population of subjects.
    '''

    for subject in subject_list:
        print(f'========== working on {subject} ==========')
        create_stc_epochs(subject)
        split_stcs_into_cond(subject)
        average_stcs(subject)
        plot_n2pc_stc(subject)

def grand_average_stc(subject_list):

    input_dir, o = get_paths()

    if not os.path.exists(os.path.join(input_dir, 'all_subj', 'stc_GA', 'plots')):
        os.makedirs(os.path.join(input_dir, 'all_subj', 'stc_GA', 'plots'))
    if not os.path.exists(os.path.join(input_dir, 'all_subj', 'stc_GA', 'data')):
        os.makedirs(os.path.join(input_dir, 'all_subj', 'stc_GA', 'data'))

    data_list = list()
    for subject in subject_list:
        cond_data = load_evoked_stc(subject)
        data_list.append(cond_data)

    grand_average = dict()
    for condition in cond_data.keys():
        grand_average[condition] = np.mean([data[condition] for data in data_list], axis=0)
        np.save(os.path.join(input_dir, 'all_subj', 'stc_GA', 'data', f'grand_average_{condition}.npy'), grand_average[condition])
        print(f'========== grand average {condition} saved ==========')

        # plot the mean time course for the labels
        inferiorparietal_lh_ts = grand_average[condition][14]
        inferiorparietal_rh_ts = grand_average[condition][15]
        lateraloccipital_lh_ts = grand_average[condition][22]
        lateraloccipital_rh_ts = grand_average[condition][23]

        # get the time points for the mean time course
        times = np.linspace(-200, 800, 512)

        # plot the mean time course for the labels
        plt.plot(times, inferiorparietal_lh_ts, label='inferiorparietal_lh')
        plt.plot(times, inferiorparietal_rh_ts, label='inferiorparietal_rh')
        plt.plot(times, lateraloccipital_lh_ts, label='lateraloccipital_lh')
        plt.plot(times, lateraloccipital_rh_ts, label='lateraloccipital_rh')
        plt.legend()
        plt.title(f'grand average {condition}')
        plt.savefig(os.path.join(input_dir, 'all_subj', 'stc_GA', 'plots', f'grand_average_{condition}.png'))
        plt.close()

    pass

###################################################################################################


full_subject_list = ['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23','70', '71', '72',
                      '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87','52', '54', '55', '56', '58','51', '53', '59', '60']


for subject in full_subject_list:
    create_stc_epochs(subject)


#subject_list = ['70', '71', '72', '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87']
