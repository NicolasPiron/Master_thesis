import os
import pandas as pd
from resting_func.set_paths import get_paths

def get_sj_conn(subject_id, input_dir):
    conn_dfs = []
    conditions = {'RESTINGSTATEOPEN': 'open', 'RESTINGSTATECLOSE': 'closed'}
    levels = ['sensor', 'source']
    freq_bands = ['theta', 'alpha', 'low_beta', 'high_beta']
    
    group_mapping = {id: 'pulvinar' for id in [51, 53, 59, 60]}
    group_mapping.update({id: 'thalamus' for id in [52, 54, 55, 56, 58]})
    subject_id_int = int(subject_id)
    group = group_mapping.get(subject_id_int, 'old' if subject_id_int < 50 else 'young' if subject_id_int > 69 else None)

    for cond, cond_name in conditions.items():
        for level in levels:
            for freq_band in freq_bands:
                if level == 'sensor':
                    file_path = os.path.join(input_dir, f'sub-{subject_id}', cond, 'connectivity', 'dynamic',
                                            f'{level}-level', 'metrics', f'sub-{subject_id}-{cond}-{freq_band}-ciplv-global-conn-metrics.csv')
                elif level == 'source':
                    file_path = os.path.join(input_dir, f'sub-{subject_id}', cond, 'connectivity', 'dynamic',
                                            f'{level}-level', 'metrics', f'sub-{subject_id}-{cond}-{freq_band}-ciplv-fpdy-conn-metrics.csv')
                try:
                    df = pd.read_csv(file_path)
                    df.insert(0, "ID", subject_id)
                    df.insert(1, "group", group)
                    df['freq'] = freq_band
                    df['level'] = level
                    df['eyes'] = cond_name
                    conn_dfs.append(df)
                except FileNotFoundError as e:
                    print(f'File not found: {file_path}')
    
    if conn_dfs:
        df = pd.concat(conn_dfs)
        df_pivoted = df.pivot_table(index=['ID', 'group', 'freq', 'level', 'eyes'],
                                           columns='metric', values='ciplv').reset_index()
        df_pivoted['range'] = df_pivoted['max'] - df_pivoted['min']
        df_pivoted.rename_axis('index', axis='columns', inplace=True)
        return df_pivoted
    else:
        return pd.DataFrame()  

def get_conn_dataset(input_dir):
    
    df_list=[]
    for directory in sorted(os.listdir(input_dir)):
        if 'sub-' in directory:
            subject_id = directory.split('-')[-1]
            df_list.append(get_sj_conn(subject_id, input_dir))
    
    df = pd.concat(df_list)
    
    if not os.path.exists(os.path.join(input_dir, 'all_subj', 'resting-state', 'connectivity', 'dynamic', 'df')):
        os.makedirs(os.path.join(input_dir, 'all_subj', 'resting-state', 'connectivity', 'dynamic', 'df'))
    df.to_csv(os.path.join(input_dir, 'all_subj', 'resting-state', 'connectivity', 'dynamic', 'df', 'all_subj_conn_ciplv.csv'), index=False)
    
    return df

def get_sj_global_src_conn(subject_id, input_dir):
    conn_dfs = []
    conditions = {'RESTINGSTATEOPEN': 'open', 'RESTINGSTATECLOSE': 'closed'}
    freq_bands = ['theta', 'alpha', 'low_beta', 'high_beta']
    
    group_mapping = {id: 'pulvinar' for id in [51, 53, 59, 60]}
    group_mapping.update({id: 'thalamus' for id in [52, 54, 55, 56, 58]})
    subject_id_int = int(subject_id)
    group = group_mapping.get(subject_id_int, 'old' if subject_id_int < 50 else 'young' if subject_id_int > 69 else None)

    for cond, cond_name in conditions.items():
            for freq_band in freq_bands:
                file_path = os.path.join(input_dir, f'sub-{subject_id}', cond, 'connectivity', 'dynamic',
                                         f'source-level', 'metrics', f'sub-{subject_id}-{cond}-{freq_band}-ciplv-global-conn-metrics.csv')
                try:
                    df = pd.read_csv(file_path)
                    df.insert(0, "ID", subject_id)
                    df.insert(1, "group", group)
                    df['freq'] = freq_band
                    df['eyes'] = cond_name
                    conn_dfs.append(df)
                except FileNotFoundError as e:
                    print(f'File not found: {file_path}')
    
    if conn_dfs:
        df = pd.concat(conn_dfs)
        df_pivoted = df.pivot_table(index=['ID', 'group', 'freq', 'eyes'],
                                           columns='metric', values='ciplv').reset_index()
        df_pivoted['range'] = df_pivoted['max'] - df_pivoted['min']
        df_pivoted.rename_axis('index', axis='columns', inplace=True)
        return df_pivoted
    else:
        return pd.DataFrame()  

def get_global_src_conn_dataset(input_dir):
    df_list=[]
    for directory in sorted(os.listdir(input_dir)):
        if 'sub-' in directory:
            subject_id = directory.split('-')[-1]
            df_list.append(get_sj_global_src_conn(subject_id, input_dir))
    
    df = pd.concat(df_list)
    
    if not os.path.exists(os.path.join(input_dir, 'all_subj', 'resting-state', 'connectivity', 'dynamic', 'df')):
        os.makedirs(os.path.join(input_dir, 'all_subj', 'resting-state', 'connectivity', 'dynamic', 'df'))
    df.to_csv(os.path.join(input_dir, 'all_subj', 'resting-state', 'connectivity', 'dynamic', 'df', 'all_subj_global_src_conn_ciplv.csv'), index=False)
    
    return df

def get_sj_hemi_src_conn(subject_id, input_dir):
    conn_dfs = []
    conditions = {'RESTINGSTATEOPEN': 'open', 'RESTINGSTATECLOSE': 'closed'}
    freq_bands = ['theta', 'alpha', 'low_beta', 'high_beta']
    
    group_mapping = {id: 'pulvinar' for id in [51, 53, 59, 60]}
    group_mapping.update({id: 'thalamus' for id in [52, 54, 55, 56, 58]})
    subject_id_int = int(subject_id)
    group = group_mapping.get(subject_id_int, 'old' if subject_id_int < 50 else 'young' if subject_id_int > 69 else None)

    for cond, cond_name in conditions.items():
        for freq_band in freq_bands:
            file_path = os.path.join(input_dir, f'sub-{subject_id}', cond, 'connectivity', 'dynamic',
                                         f'source-level', 'metrics', f'sub-{subject_id}-{cond}-{freq_band}-ciplv-hemi-conn-metrics.csv')
            try:
                df = pd.read_csv(file_path)
                df['side'] = df['metric'].apply(lambda x: 'left' if x.startswith('left_') else 'right')
                df['metric'] = df['metric'].str.replace('left_', '').str.replace('right_', '')
                df.insert(0, "ID", subject_id)
                df.insert(1, "group", group)
                df['freq'] = freq_band
                df['eyes'] = cond_name
                conn_dfs.append(df)
            except FileNotFoundError as e:
                print(f'File not found: {file_path}')

    if conn_dfs:
        return pd.concat(conn_dfs)
    else:
        return pd.DataFrame()

def get_hemi_src_conn_dataset(input_dir):
    df_list=[]
    for directory in sorted(os.listdir(input_dir)):
        if 'sub-' in directory:
            subject_id = directory.split('-')[-1]
            df_list.append(get_sj_hemi_src_conn(subject_id, input_dir))
    
    df = pd.concat(df_list)
    
    if not os.path.exists(os.path.join(input_dir, 'all_subj', 'resting-state', 'connectivity', 'dynamic', 'df')):
        os.makedirs(os.path.join(input_dir, 'all_subj', 'resting-state', 'connectivity', 'dynamic', 'df'))
    df.to_csv(os.path.join(input_dir, 'all_subj', 'resting-state', 'connectivity', 'dynamic', 'df', 'all_subj_hemi_src_conn_ciplv.csv'), index=False)
    
    return df

def get_sj_power(subject_id, input_dir):

    power_dfs = []
    conditions = {'RESTINGSTATEOPEN': 'open', 'RESTINGSTATECLOSE': 'closed'}
    levels = ['sensor', 'source']
   
    group_mapping = {id: 'pulvinar' for id in [51, 53, 59, 60]}
    group_mapping.update({id: 'thalamus' for id in [52, 54, 55, 56, 58]})
    subject_id_int = int(subject_id)
    group = group_mapping.get(subject_id_int, 'old' if subject_id_int < 50 else 'young' if subject_id_int > 69 else None)

    for cond, cond_name in conditions.items():
        for level in levels:
            file_path = os.path.join(input_dir, f'sub-{subject_id}', cond, 'psd',
                                        f'{level}-level', 'power-df', f'sub-{subject_id}-{cond}-power.csv')
            try:
                df = pd.read_csv(file_path)
                df.insert(0, "ID", subject_id)
                df.insert(1, "group", group)
                df['level'] = level
                df['eyes'] = cond_name
                df.rename(columns = {'Unnamed: 0':'roi'}, inplace = True) 
                power_dfs.append(df)
            except FileNotFoundError as e:
                print(f'File not found: {file_path}')

    if power_dfs:
        df = pd.concat(power_dfs)
        return df
    else:
        return pd.DataFrame()

def get_power_dataset(input_dir):
    
    df_list=[]
    for directory in sorted(os.listdir(input_dir)):
        if 'sub-' in directory:
            subject_id = directory.split('-')[-1]
            df_list.append(get_sj_power(subject_id, input_dir))
    
    df = pd.concat(df_list)
    
    if not os.path.exists(os.path.join(input_dir, 'all_subj', 'resting-state', 'power', 'df')):
        os.makedirs(os.path.join(input_dir, 'all_subj', 'resting-state', 'power', 'df'))
    df.to_csv(os.path.join(input_dir, 'all_subj', 'resting-state', 'power', 'df', 'last_rs_all_subj_power.csv'), index=False)
    
    return df

if __name__ == '__main__':

    i, o = get_paths()
    get_conn_dataset(i)
    get_global_src_conn_dataset(i)
    get_hemi_src_conn_dataset(i)
    #get_power_dataset(i)