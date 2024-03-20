import pandas as pd
import numpy as np
import os

def get_rs_PO_alpha(subject_id, input_dir):
    
    rs_closed_psd = np.load(os.path.join(input_dir, f'sub-{subject_id}/RESTINGSTATECLOSE/psd/sensor-level',
                                        f'psd-data/spectrum/sub-{subject_id}-RESTINGSTATECLOSE-psd.npy'))
    rs_open_psd = np.load(os.path.join(input_dir, f'sub-{subject_id}/RESTINGSTATEOPEN/psd/sensor-level', 
                                       f'psd-data/spectrum/sub-{subject_id}-RESTINGSTATEOPEN-psd.npy'))
    
    po8_closed = rs_closed_psd[61,:]
    po8_open = rs_open_psd[61,:]
    po8_closed_alpha = po8_closed[80:120].mean(axis=0)
    po8_open_alpha = po8_open[80:120].mean(axis=0)
    
    po7_closed = rs_closed_psd[24,:]
    po7_open = rs_open_psd[24,:]
    po7_closed_alpha = po7_closed[80:120].mean(axis=0)
    po7_open_alpha = po7_open[80:120].mean(axis=0)
    
    open_dict = {}
    closed_dict = {}
    
    open_dict['alpha-PO7'] = po7_open_alpha
    open_dict['alpha-PO8'] = po8_open_alpha
    closed_dict['alpha-PO7'] = po7_closed_alpha
    closed_dict['alpha-PO8'] = po8_closed_alpha
    
    return open_dict, closed_dict

def get_all_subj_rs_po_alpha(subject_list, input_dir):
    
    rs_open_data = {}
    rs_closed_data = {}
    
    for subject_id in subject_list:
        try:
            open_dict, closed_dict = get_rs_PO_alpha(subject_id, input_dir)
            rs_open_data[subject_id] = open_dict
            rs_closed_data[subject_id] = closed_dict
        except:
            print(f'did not found data for {subject_id}')

    return rs_open_data, rs_closed_data

        
def create_rows_from_dict(data_dict, condition):
    rows = []
    for id, values in data_dict.items():
        row = {'ID': int(id), 'condition': condition, 'alpha-PO7': values['alpha-PO7'],
               'alpha-PO8': values['alpha-PO8']}
        rows.append(row)
    return rows

def add_rs_data(df, rs_open_data, rs_closed_data):

    rs_open_rows = create_rows_from_dict(rs_open_data, 'RS_open')
    rs_close_rows = create_rows_from_dict(rs_closed_data, 'RS_closed')
    new_rows_df = pd.DataFrame(rs_open_rows + rs_close_rows)
    extended_df = pd.concat([df, new_rows_df], ignore_index=True)
    
    return extended_df

def main(input_dir):
    
    subject_list = os.listdir(input_dir)
    subject_list = [sub[-2:] for sub in subject_list if sub.startswith('sub-')]
    subject_list = sorted(subject_list)
    
    orig_df = pd.read_csv(os.path.join(input_dir, 'all_subj', 'N2pc', 'n2pc-values',
                                       'n2pc-values-around-peak', 'all_subjects_amp_power.csv'), index_col=0)
    power_df = orig_df[['ID', 'condition', 'alpha-PO7', 'alpha-PO8']]
    averaged_df = power_df.groupby(['ID', 'condition']).mean().reset_index()
    
    rs_open_data, rs_closed_data = get_all_subj_rs_po_alpha(subject_list, input_dir)
    df = add_rs_data(averaged_df, rs_open_data, rs_closed_data)
    
    # invert left and right for the patients that have a lesion on the right 
    # -> the channels are positionned the same way to the lesion for each patient. 
    subject_ids = [51, 53, 54, 58, 59]
    subset_df = df[df['ID'].isin(subject_ids)]
    subset_df[['alpha-PO7', 'alpha-PO8']] = subset_df[['alpha-PO8', 'alpha-PO7']]
    df.update(subset_df)
    
    grp_mapping = {'old':list(range(1, 24)),
              'pulvinar':[51, 53, 59, 60],
              'thalamus':[52, 54, 55, 56, 57, 58],
              'young':list(range(70, 88))}
    id_to_group = {id_: group for group, ids in grp_mapping.items() for id_ in ids}
    df['group'] = df['ID'].map(id_to_group)
    
    df.to_csv(os.path.join(input_dir, 'all_subj', 'N2pc', 'alpha-power-allsubj', 'n2pc_RS_alpha_average.csv'))

