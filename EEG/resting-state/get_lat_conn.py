import pandas as pd
import os
import numpy as np
from resting_func.set_paths import get_paths

def get_df(mat):

    long_df = mat.reset_index().melt(id_vars='index', var_name='col', value_name='ciPLV')
    long_df_sorted = long_df.sort_values(by='ciPLV')
    long_df_sorted = long_df_sorted[long_df_sorted['ciPLV'] != 0]
    long_df_sorted.reset_index(drop=True, inplace=True)
    long_df_sorted['hemi'] = np.nan * len(long_df_sorted)
    return long_df_sorted

def get_hemi(row):

    last_1 = row['index'][-1]
    last_2 = row['col'][-1]
    if last_1 == 'z' or last_2 == 'z':
        pass
    elif int(last_1)%2 == 0 and int(last_2)%2 == 0:
        return 'right'
    elif int(last_1)%2 != 0 and int(last_2)%2 != 0:
        return 'left'
    elif (int(last_1)%2 != 0 and int(last_2)%2 == 0) or (int(last_1)%2 == 0 and int(last_2)%2 != 0):
        return 'inter'

def average_conn_subj(data_path, subject_id):

    mat = pd.read_csv(data_path, index_col=0)
    long_df_sorted = get_df(mat)
    long_df_sorted['hemi'] = long_df_sorted.apply(get_hemi, axis=1)
    grouped = long_df_sorted.groupby('hemi')['ciPLV'].mean()
    grouped = grouped.reset_index()
    grouped.columns = ['hemi', 'ciPLV']
    grouped.loc[len(grouped.index)] = ['all', long_df_sorted['ciPLV'].mean()]
    grouped['ID'] = subject_id
    return grouped

def get_average_conn(subject_list, freq_name, freq_range, condition):

    i, _ = get_paths()

    if condition == 'closed':
        rs_dir = 'RESTINGSTATECLOSE'
    elif condition == 'open':
        rs_dir = 'RESTINGSTATEOPEN'

    lower = freq_range[0]
    upper = freq_range[1]
    df_list = []

    for subject in subject_list:
        data_path = os.path.join(i, f'sub-{subject}',rs_dir, 'connectivity', 'static', 'conn_data',
                                 f'sub-{subject}-static-ciplv-{lower}-{upper}-{condition}.csv')
        df = average_conn_subj(data_path, subject)
        df_list.append(df)

    df = pd.concat(df_list)

    old = list(range(1, 24))
    old = [str(i).zfill(2) for i in old]
    grp_mapping = {'old':old,
              'pulvinar':['51', '53', '59', '60'],
              'thalamus':['52', '54', '55', '56', '57', '58']}
    id_to_group = {id_: group for group, ids in grp_mapping.items() for id_ in ids}
    df['group'] = df['ID'].map(id_to_group)
    df['freq'] = freq_name
    df['eyes'] = condition

    return df

def main():

    _, o = get_paths()
    dir_name = os.path.join(o, 'all_subj', 'resting-state', 'connectivity',
                        'static', 'average')
    fname = os.path.join(dir_name, 'average_conn.csv')

    subject_list = ['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23', '51', '52', '53',
                        '54', '55', '56', '58', '59', '60']
    freq_dict = {'theta':(4, 8),
        'alpha':(8, 12),
        'low_beta':(12, 16),
        'high_beta':(16, 30)}
    conditions = ['closed', 'open']
    
    df_list = []
    for freq_name, freq_range in freq_dict.items():
        for condition in conditions:
            df = get_average_conn(subject_list, freq_name, freq_range, condition)
            df_list.append(df)
        
    df = pd.concat(df_list)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    df.to_csv(fname, index=False)

if __name__ == '__main__':

    main()