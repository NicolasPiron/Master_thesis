import numpy as np
import pandas as pd
import os
import mne
from mne_connectivity import spectral_connectivity_time
import matplotlib.pyplot as plt
import seaborn as sns
from n2pc_func.params import ch_names

def get_hemi_df(subject_id: str, input_dir: str)-> pd.DataFrame:
    ''' Get hemisphere dataframe for a given subject. If the dataframe has not been computed yet,
    it will be computed and saved.'''
    fn = os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'sensor-connectivity', f'sub-{subject_id}-hemi_df.csv')
    if not os.path.exists(fn):
        mat = get_conn_mat(subject_id, input_dir)
        long_df = mat2df(mat)
        long_df['hemi'] = long_df.apply(get_hemi, axis=1)
        long_df['ID'] = subject_id
        long_df.to_csv(fn)
    else:
        long_df = pd.read_csv(fn, index_col=0)
    return long_df

def get_hemi(row: pd.DataFrame)-> str:
    ''' Stolen from resting-state/get_lat_conn.py.
    Get hemisphere of a pair of channels. Left, right, interhemispheric or None if central electrode.'''
    last_1 = row['index'][-1]
    last_2 = row['col'][-1]
    if last_1 == 'z' or last_2 == 'z':
        return None
    elif int(last_1)%2 == 0 and int(last_2)%2 == 0:
        return 'right'
    elif int(last_1)%2 != 0 and int(last_2)%2 != 0:
        return 'left'
    elif (int(last_1)%2 != 0 and int(last_2)%2 == 0) or (int(last_1)%2 == 0 and int(last_2)%2 != 0):
        return 'inter'

def mat2df(mat: pd.DataFrame)-> pd.DataFrame:
    '''Stolen from resting-state/get_lat_conn.py (named get_df()).
    Get a long dataframe from a connectivity matrix.'''
    long_df = mat.reset_index().melt(id_vars='index', var_name='col', value_name='ciPLV')
    print(long_df)
    long_df_sorted = long_df.sort_values(by='ciPLV')
    long_df_sorted = long_df_sorted[long_df_sorted['ciPLV'] != 0]
    long_df_sorted.reset_index(drop=True, inplace=True)
    long_df_sorted['hemi'] = np.nan * len(long_df_sorted)
    return long_df_sorted

def plot_mat(data, title, ch_names, vmin=-1, vmax=1)-> plt.figure: 
    ''' Plots a matrix'''
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(data, ax=ax, cmap='coolwarm', center=0,
                xticklabels=ch_names, yticklabels=ch_names,
                vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    plt.tight_layout()
    plt.show()
    return fig

def get_conn_mat(subject_id: str, input_dir: str)-> pd.DataFrame:
    ''' Get connectivity matrix for a given subject. If the matrix has not been computed yet,
    it will be computed and saved.'''
    conn_dir = os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'sensor-connectivity')
    if not os.path.exists(conn_dir):
        os.makedirs(conn_dir)
    conn_fn = os.path.join(conn_dir, f'sub-{subject_id}-connectivity.csv')
    if not os.path.exists(conn_fn):
        epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))
        data = epochs.get_data()[:, :64, :]
        con_mat = cmpt_conn_mat(data)
        pd.DataFrame(con_mat, index=ch_names, columns=ch_names).to_csv(conn_fn)
    else:
        con_mat = pd.read_csv(conn_fn, index_col=0)
    return con_mat

def cmpt_conn_mat(epochs:np.array)-> np.array:
    ''' Compute connectivity matrix from epochs data
    
    Parameters
    ----------
    epochs : np.array
        Epochs data
    
    Returns
    -------
    con_mat : np.array
        Connectivity matrix
    '''
    fmin = 8.0
    fmax = 13.0
    freqs = np.arange(fmin, fmax)
    sfreq = 512
    n_chan = epochs.shape[1]
    con = spectral_connectivity_time(
        epochs,
        method='ciplv',
        average=True,
        mode="multitaper",
        sfreq=sfreq,
        freqs=freqs,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        n_jobs=1,
    )

    return con.get_data().reshape(n_chan, n_chan)


