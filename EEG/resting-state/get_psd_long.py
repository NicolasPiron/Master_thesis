import os
import numpy as np
import pandas as pd
from resting_func.set_paths import get_paths
from resting_func.set_subject_lists import get_subject_list
from resting_func.rs_frequency import get_data, get_roi_dict
import seaborn as sns
import matplotlib.pyplot as plt

def get_long_subject(subject_id, swap_sides=False):

    df_list = []
    roi_dict = get_roi_dict()
    conditions = ['RESTINGSTATEOPEN', 'RESTINGSTATECLOSE']

    for condition in conditions:

        if condition == 'RESTINGSTATEOPEN':
            cond_name = 'open'
        else:
            cond_name = 'closed'

        spec, freq = get_data(subject_id, condition)

        data_dict = {}
        for roi in roi_dict.keys():
            data_dict[roi] = spec[roi_dict[roi]].mean(axis=0)

        df = pd.DataFrame(data_dict)
        df['freq'] = freq
        df['condition'] = cond_name
        df['ID'] = subject_id
        df_long = pd.melt(df, id_vars=['ID', 'condition', 'freq'], var_name='ROI', value_name='PSD')
        df_list.append(df_long)

    df = pd.concat(df_list, axis=0)

    def swap_suffix(roi):
        if roi.endswith('_l'):
            return roi[:-2] + '_r'
        elif roi.endswith('_r'):
            return roi[:-2] + '_l'
        return roi

    if swap_sides:

        df['ROI'] = df['ROI'].apply(swap_suffix)

    return df

def get_long_all_subjects(subject_list):

    to_be_swapped = ['51', '53', '54', '58', '59']

    _, o = get_paths()
    path = os.path.join(o, 'all_subj', 'resting-state', 'power', 'df')
    if not os.path.exists(path):
        os.makedirs(path)

    df_list = []

    for subject in subject_list:
        if subject in to_be_swapped:
            swap_sides = True
        else:
            swap_sides = False
        try:
            df = get_long_subject(subject, swap_sides=swap_sides)
            df_list.append(df)
        except:
            print(f'========== Error with subject {subject} ==========')

    df = pd.concat(df_list, axis=0)
    df.to_csv(os.path.join(path, 'long_psd_corrected.csv'), index=False)

    return df

if __name__ == '__main__':

    subject_list = get_subject_list()
    df = get_long_all_subjects(subject_list)
    subject_in_df = df['ID'].unique()
    print('========== Done ==========')
    print(f'subjects in df : {subject_in_df}')



