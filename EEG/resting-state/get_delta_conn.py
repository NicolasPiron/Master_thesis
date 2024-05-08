import pandas as pd
import os
import numpy as np
from resting_func.set_paths import get_paths


def multiply_by_adjacency_matrix(df, adjacency_matrix):

    df = df.copy()
    ones = adjacency_matrix == 1
    number_of_ones = ones.sum().sum()
    print('Multiplying connectivity matrix by adjacency matrix')
    print(f'{number_of_ones} significant edges')
    for i in range(len(df)):
        for j in range(len(df)):
            df.iloc[i, j] = df.iloc[i, j] * adjacency_matrix.iloc[i, j]
    print('Connectivity matrix multiplied by adjacency matrix')
    return df

def extract_percentiles(df, n):

    long_df = df.reset_index().melt(id_vars='index', var_name='col', value_name='ciPLV')
    long_df_sorted = long_df.sort_values(by='ciPLV')
    long_df_sorted = long_df_sorted[long_df_sorted['ciPLV'] != 0]
    # Extract top and bottom percentile values
    bottom_percentile_indices = long_df_sorted.head(n)
    top_percentile_indices = long_df_sorted.tail(n)
    return bottom_percentile_indices, top_percentile_indices

def get_vals_subj(subject_id, group1, group2, freq_range, freq_name, condition):

    i, _ = get_paths()

    if condition == 'closed':
        rs_dir = 'RESTINGSTATECLOSE'
    elif condition == 'open':
        rs_dir = 'RESTINGSTATEOPEN'

    lower = freq_range[0]
    upper = freq_range[1]
    
    comparison_path = os.path.join(i, 'all_subj', 'resting-state', 'connectivity', 'static', 'clean_nbs_results')
    subj_mat_path = os.path.join(i, f'sub-{subject_id}',rs_dir, 'connectivity', 'static', 'conn_data',
                                 f'sub-{subject_id}-static-ciplv-{lower}-{upper}-{condition}.csv')
    if '_' in group1:
        group1 = group1.split('_')[0]
    if '_' in group2:
        group2 = group2.split('_')[0]
    comp_name = f'{group1}-{freq_name}-ciplv-{condition}_VS_{group2}-{freq_name}-ciplv-{condition}_07'
    adj_mat_path = os.path.join(comparison_path, 'clean_nbs_results', comp_name, 'adj.csv')
    
    sub_mat = pd.read_csv(subj_mat_path, index_col=0)
    adj_mat = pd.read_csv(adj_mat_path, index_col=0)

    # 10% of the edges
    n_edges = (adj_mat == 1).sum().sum()
    n = n_edges // 10

    sub_sign_mat = multiply_by_adjacency_matrix(sub_mat, adj_mat)
    bottom_percentile_values, top_percentile_values = extract_percentiles(sub_sign_mat, n)

    # merge the two dataframes
    bottom_percentile_values['percentile'] = 'bottom'
    top_percentile_values['percentile'] = 'top'
    df = pd.concat([bottom_percentile_values, top_percentile_values])
    df['ID'] = subject_id
    df['comparison'] = f'{group1} VS {group2}'
    df['freq'] = freq_name
    df['eyes'] = condition

    return df

def get_hemi(row):

    last_1 = row['index'][-1]
    last_2 = row['col'][-1]
    if last_1 == 'z' or last_2 == 'z':
        if last_1 == 'z' and last_2 == 'z':
            pass
        elif last_1 == 'z':
            if int(last_2)%2 == 0:
                return 'right'
            elif int(last_2)%2 != 0:
                return 'left'
        elif last_2 == 'z':
            if int(last_1)%2 == 0:
                return 'right'
            elif int(last_1)%2 != 0:
                return 'left'
    elif int(last_1)%2 == 0 and int(last_2)%2 == 0:
        return 'right'
    elif int(last_1)%2 != 0 and int(last_2)%2 != 0:
        return 'left'
    elif (int(last_1)%2 != 0 and int(last_2)%2 == 0) or (int(last_1)%2 == 0 and int(last_2)%2 != 0):
        return 'inter'
    
def get_cluster_count(df):

    temporary_store = pd.DataFrame(data={'cluster':2*['frontal', 'central', 'parietal', 'occipital'],
                                        'hemi':['left']*4 + ['right']*4, 'n':np.zeros(8)})   
    sub_df = df[(df['hemi'] != 'inter')]
    for i, row in sub_df.iterrows():
        n1 = row['index']
        n2 = row['col']
        hemi = row['hemi']
        if 'F' in n1 or 'F' in n2:
            temporary_store.loc[(temporary_store['cluster'] == 'frontal') & (temporary_store['hemi'] == hemi), 'n'] += 1
        if 'C' in n1 or 'C' in n2:
            temporary_store.loc[(temporary_store['cluster'] == 'central') & (temporary_store['hemi'] == hemi), 'n'] += 1
        if 'P' in n1 or 'P' in n2:
            temporary_store.loc[(temporary_store['cluster'] == 'parietal') & (temporary_store['hemi'] == hemi), 'n'] += 1
        if 'O' in n1 or 'O' in n2:
            temporary_store.loc[(temporary_store['cluster'] == 'occipital') & (temporary_store['hemi'] == hemi), 'n'] += 1
    return temporary_store

def get_cluster_delta(temporary_store):

    delta_df = pd.DataFrame(data={'cluster':['frontal', 'central', 'parietal', 'occipital'], 'delta':np.zeros(4)})
    
    frontal = temporary_store[temporary_store['cluster'] == 'frontal']
    central = temporary_store[temporary_store['cluster'] == 'central']
    parietal = temporary_store[temporary_store['cluster'] == 'parietal']
    occipital = temporary_store[temporary_store['cluster'] == 'occipital']
    
    frontal_delta = frontal.loc[frontal['hemi'] == 'left', 'n'].values[0] - frontal.loc[frontal['hemi'] == 'right', 'n'].values[0]
    central_delta = central.loc[central['hemi'] == 'left', 'n'].values[0] - central.loc[central['hemi'] == 'right', 'n'].values[0]
    parietal_delta = parietal.loc[parietal['hemi'] == 'left', 'n'].values[0] - parietal.loc[parietal['hemi'] == 'right', 'n'].values[0]
    occipital_delta = occipital.loc[occipital['hemi'] == 'left', 'n'].values[0] - occipital.loc[occipital['hemi'] == 'right', 'n'].values[0]

    delta_df.loc[delta_df['cluster'] == 'frontal', 'delta'] = frontal_delta
    delta_df.loc[delta_df['cluster'] == 'central', 'delta'] = central_delta
    delta_df.loc[delta_df['cluster'] == 'parietal', 'delta'] = parietal_delta
    delta_df.loc[delta_df['cluster'] == 'occipital', 'delta'] = occipital_delta
    
    return delta_df

def get_delta_subject(subject_id, group1, group2, freq_range, freq_name, condition):

    df = get_vals_subj(subject_id, group1, group2, freq_range, freq_name, condition)
    df['hemi'] = df.apply(get_hemi, axis=1)

    df_list = []
    for percentile in ['top', 'bottom']:
        cluster_count = get_cluster_count(df[df['percentile'] == percentile])
        delta_df = get_cluster_delta(cluster_count)
        delta_df['percentile'] = percentile
        df_list.append(delta_df)

    delta_df = pd.concat(df_list)
    delta_df['ID'] = subject_id
    delta_df['comparison'] = f'{group1} VS {group2}'
    delta_df['freq'] = freq_name
    delta_df['eyes'] = condition
    return delta_df

def main():

    _, o = get_paths()
    dir_name = os.path.join(o, 'all_subj', 'resting-state', 'connectivity','static', 'average')
    fname = os.path.join(dir_name, 'average_delta_conn.csv')
    
    subject_dict = {'old': ['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23'], 'thal':['52',
                        '54', '55', '56', '58'], 'pulvinar':['51', '53', '59', '60']}

    mapping = {1:{'freq_name':'alpha','condition':'open'},
            2:{'freq_name':'alpha','condition':'closed'},
            3:{'freq_name':'theta','condition':'open'},
            4:{'freq_name':'theta','condition':'closed'},
            5:{'freq_name':'low_beta','condition':'closed'}}

    comparisons = {('old', 'pulvinar'):[1, 2, 3, 4, 5],
                ('old', 'thal'):[3, 4],
                ('thal', 'pulvinar'):[2, 5]}

    freq_dict = {'theta':(4, 8),
        'alpha':(8, 12),
        'low_beta':(12, 16)}

    df_list = []
    for group1, group2 in comparisons.keys():
        for combinaison in comparisons[(group1, group2)]:
            freq_name = mapping[combinaison]['freq_name']
            freq_range = freq_dict[freq_name]
            condition = mapping[combinaison]['condition']
            for subject in subject_dict[group1]:
                df = get_delta_subject(subject, group1, group2, freq_range, freq_name, condition)
                df_list.append(df)
    df = pd.concat(df_list)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    df.to_csv(fname, index=False)
