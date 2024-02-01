import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from set_paths import get_paths
from mne.viz import circular_layout
from mne_connectivity import spectral_connectivity_time
from mne_connectivity.viz import plot_connectivity_circle
from mne import read_labels_from_annot
from mne.datasets import fetch_fsaverage
from src_rec import combine_conditions

###################################################################################################
# Basic connectivity functions
###################################################################################################

def get_connectivity_matrix(data):

    fmin = 8.0
    fmax = 13.0
    freqs = np.arange(fmin, fmax)
    sfreq = 512
    con = spectral_connectivity_time(
        data,
        method='plv',
        average=True,
        mode="multitaper",
        sfreq=sfreq,
        freqs=freqs,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        n_jobs=1,
    )
    con_mat=con.get_data().reshape(int(np.sqrt(4624)), int(np.sqrt(4624)))

    return con_mat

def get_labels(return_names=False):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if 'nicolaspiron/Documents' in script_dir:
        label_path = fetch_fsaverage(verbose=True)
    elif 'shared_PULSATION' in script_dir:
        label_path = '/home/nicolasp/shared_PULSATION/MNE-fsaverage-data/fsaverage'
    else:
        raise Exception('Please specify the path to the fsaverage directory in the source_set_up function.') 
    labels = read_labels_from_annot('', parc='aparc', subjects_dir=label_path)
    labels = [label for label in labels if 'unknown' not in label.name]

    if return_names:
        label_names = [label.name for label in labels]
        return labels, label_names
    else:   
        return labels

def save_con_mat(con_mat, path, name):

    _, label_names = get_labels(return_names=True)
    df = pd.DataFrame(con_mat, index=label_names, columns=label_names)
    df.to_csv(f'{path}/{name}.csv')

def load_con_mat(path, name):

    df = pd.read_csv(f'{path}/{name}.csv', index_col=0)
    con_mat = df.to_numpy()
    return con_mat

def plot_con_mat(con_mat, title):

    _, label_names = get_labels(return_names=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(title)
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=90, fontsize=8)
    ax.set_yticklabels(label_names, fontsize=8)
    im0 = ax.imshow(con_mat)
    fig.colorbar(im0, shrink=0.81)
    fig.tight_layout()

    return fig

def plot_con_circle(con_mat):

    labels, label_names = get_labels(return_names=True)
    lh_labels = [name for name in label_names if name.endswith("lh")]

    # Get the y-location of the label
    label_ypos = list()
    for name in lh_labels:
        idx = label_names.index(name)
        ypos = np.mean(labels[idx].pos[:, 1])
        label_ypos.append(ypos)
        
    # Reorder the labels based on their location
    lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]
    # For the right hemi
    rh_labels = [label[:-2] + "rh" for label in lh_labels]
    # Save the plot order and create a circular layout
    node_order = list()
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)
    node_angles = circular_layout(
        label_names, node_order, start_pos=90, group_boundaries=[0, len(label_names) / 2]
    )
    label_colors = [label.color for label in labels]

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True))
    plot_connectivity_circle(
        con_mat,
        label_names,
        n_lines=300,
        node_angles=node_angles,
        node_colors=label_colors,
        ax=ax,
        show=False,
    )
    fig.tight_layout()

    return fig

###################################################################################################
# Task-specific connectivity functions
###################################################################################################

def con_pipeline_single_subj(subject_id):

    input_dir, o = get_paths()
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'src-connectivity')):
        os.mkdir(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'src-connectivity'))
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'src-connectivity', 'con-matrices')):
        os.mkdir(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'src-connectivity', 'con-matrices'))
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'src-connectivity', 'con-figures')):
        os.mkdir(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'src-connectivity', 'con-figures'))

    three_cond_data = combine_conditions(subject_id)

    for condition, data in three_cond_data.items():
        con_mat = get_connectivity_matrix(data)
        save_con_mat(con_mat, os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'src-connectivity', 'con-matrices'), f'{condition}-con-mat')
        mat_fig = plot_con_mat(con_mat, f'{condition}-con-mat')
        mat_fig.savefig(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'src-connectivity', 'con-figures', f'{condition}-con-mat.png'))
        plt.close()
        circle_fig = plot_con_circle(con_mat)
        circle_fig.savefig(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'src-connectivity', 'con-figures', f'{condition}-con-circle.png'))
        plt.close()

    return None

def con_pipeline_population(subject_list, population):

    input_dir, o = get_paths()

    for path in [os.path.join(input_dir, 'all_subj', 'N2pc', 'src-connectivity', population, 'con-matrices'),
                 os.path.join(input_dir, 'all_subj', 'N2pc', 'src-connectivity', population, 'con-figures')]:
        if not os.path.exists(path):
            os.makedirs(path)

    conditions = ['dis_mid', 'dis_lat', 'no_dis']
    mat_dict = dict()
    for condition in conditions:
        mat_dict[condition] = list()
        for subject_id in subject_list:
            mat_dict[condition].append(load_con_mat(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'src-connectivity', 'con-matrices'), f'{condition}-con-mat'))
        mat_dict[condition] = np.mean(mat_dict[condition], axis=0)
        save_con_mat(mat_dict[condition], os.path.join(input_dir, 'all_subj', 'N2pc', 'src-connectivity', population, 'con-matrices'), f'{condition}-con-mat')
        mat_fig = plot_con_mat(mat_dict[condition], f'{condition}-con-mat')
        mat_fig.savefig(os.path.join(input_dir, 'all_subj', 'N2pc', 'src-connectivity', population, 'con-figures', f'{condition}-con-mat.png'))
        plt.close()
        circle_fig = plot_con_circle(mat_dict[condition])
        circle_fig.savefig(os.path.join(input_dir, 'all_subj', 'N2pc', 'src-connectivity', population, 'con-figures', f'{condition}-con-circle.png'))
        plt.close()

    return None

###################################################################################################
# Functions to add PLV values to the trial by trial dataframe
###################################################################################################

def get_ROI_con_values_epochs(subject_id):

#   'caudalmiddlefrontal_lh': 4,
#   'caudalmiddlefrontal_rh': 5,
#   'inferiorparietal_lh': 14,
#   'inferiorparietal_rh': 15

    input_dir, o = get_paths()
    data = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'stc_epochs', f'sub-{subject_id}-stc_epochs.npy'))
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'src-connectivity', 'epoch-con-matrices')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'src-connectivity', 'epoch-con-matrices'))

    fmin = 8.0
    fmax = 13.0
    freqs = np.arange(fmin, fmax)
    sfreq = 512
    indices = ([4, 4, 4, 5, 5, 14], [5, 14, 15, 14, 15, 15]) # Indices of the ROI's in the con matrix
    con = spectral_connectivity_time(
        data,
        method=['plv', 'pli'],
        mode="multitaper",
        sfreq=sfreq,
        indices=indices,
        freqs=freqs,
        faverage=True,
        n_jobs=1,
    )
    plv_data = con[0].get_data()
    pli_data = con[1].get_data()
    np.save(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'src-connectivity', 'epoch-con-matrices', 'plv_data.npy'), plv_data)
    np.save(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'src-connectivity', 'epoch-con-matrices', 'pli_data.npy'), pli_data)

    return None

def load_con_values_epochs(subject_id):

    input_dir, o = get_paths()
    plv_data = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'src-connectivity', 'epoch-con-matrices', 'plv_data.npy'))
    pli_data = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'src-connectivity', 'epoch-con-matrices', 'pli_data.npy'))

    return plv_data, pli_data

def create_ROI_con_values_df(subject_id):

    input_dir, o = get_paths()
    plv_data, pli_data = load_con_values_epochs(subject_id)
    n2pc_df = pd.read_csv(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-values', f'sub-{subject_id}-amplitude-around-peak.csv'), index_col=0)
    print(n2pc_df.head())
    # check that the number of trials is the same
    n_epochs = (n2pc_df['epoch_dropped'] == False).sum()
    print(f'Number of epochs: {n_epochs}')
    assert n_epochs == plv_data.shape[0]

    # create the dataframe with a column for each ROI pair
    df = pd.DataFrame(columns=['plv-cmflh-cmfrh', 'plv-cmflh-iplh', 'plv-cmflh-iprh', 'plv-cmfrh-iplh', 'plv-cmfrh-iprh', 'plv-iplh-iprh',
                               'pli-cmflh-cmfrh', 'pli-cmflh-iplh', 'pli-cmflh-iprh', 'pli-cmfrh-iplh', 'pli-cmfrh-iprh', 'pli-iplh-iprh'],
                               index=range(len(n2pc_df)))
    
    for i in range(n_epochs):
        if n2pc_df.loc[i, 'epoch_dropped'] == False:
            df.loc[i] = [plv_data[i, 0][0], plv_data[i, 1][0], plv_data[i, 2][0], plv_data[i, 3][0], plv_data[i, 4][0], plv_data[i, 5][0],
                        pli_data[i, 0][0], pli_data[i, 1][0], pli_data[i, 2][0], pli_data[i, 3][0], pli_data[i, 4][0], pli_data[i, 5][0]]
        else:
            pass

    print(f'========== Created PLV-PLI dataframe for subject {subject_id}')
    df.to_csv(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'src-connectivity', f'sub-{subject_id}-con-values.csv'))

    return None

def con_df_ROI_pipeline(subject_id):

    get_ROI_con_values_epochs(subject_id)
    create_ROI_con_values_df(subject_id)

    return None

###################################################################################################
# Run the pipeline
###################################################################################################


#population_dict = {'old_control': ['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23'],
#                    'thal_control': ['52', '54', '55', '56', '58'],
#                    'young_control': ['70', '71', '72', '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87'],
#                    'pulvinar': ['51', '53', '59', '60']}

full_subject_list = ['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23','70', '71', '72',
                      '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87','52', '54', '55', '56', '58','51', '53', '59', '60']

#subject_list = ['70', '71', '72', '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87']

for subject_id in full_subject_list:
    con_df_ROI_pipeline(subject_id)
#    

#for population, subject_list in population_dict.items():
 #   con_pipeline_population(subject_list, population)