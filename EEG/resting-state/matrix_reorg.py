import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
from resting_func.set_paths import get_paths


def find_significative_matrices(mat_dir):
    
    matrices = glob.glob(mat_dir + '/*sign.csv')
    mat_dict = {}
    for mat in matrices:
        mat_name = mat.split('/')[-1]
        mat_dict[mat_name] = pd.read_csv(mat, index_col=0)
    print(f'============= matrices in {mat_dir} =============')
    
    return mat_dict

def reorder_matrix(matrix, channel_order):

    # fill the upper triangle
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    # reorder the matrix
    matrix = matrix.loc[channel_order, channel_order]
    # get rid of the upper triangle
    matrix = pd.DataFrame(np.tril(matrix), index=matrix.index, columns=matrix.columns)
    print(f'============= matrix reorganized =============')

    return matrix

def plot_and_save_reorg_matrix(matrix, mat_name, mat_dir):

    chan_names = matrix.index.values
    conn_matrix = matrix.values

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(mat_name.split('.')[0])
    ax.set_xticks(np.arange(len(chan_names)))
    ax.set_yticks(np.arange(len(chan_names)))
    ax.set_xticklabels(chan_names, rotation=90, fontsize=8)
    ax.set_yticklabels(chan_names, fontsize=8)
    im0 = ax.imshow(conn_matrix, vmin=0, vmax=1)
    fig.colorbar(im0,shrink=0.81)
    fig.savefig(os.path.join(mat_dir, mat_name.split('.')[0] + '_reorg.png'), dpi=300)
    plt.close(fig)
    matrix.to_csv(os.path.join(mat_dir, mat_name.split('.')[0] + '_reorg.csv'))
    print(f'============= matrix plotted and saved =============')



def main():

    input_dir, _ = get_paths()
    input_dir = os.path.join(input_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'nbs_results')
    mat_dirs = os.listdir(input_dir)
    mat_dirs.remove('all_pvals')


    channel_order = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3','FC1', 'C1',
                 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7',
                 'PO3','O1', 'Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz','POz', 'Oz','Iz',
                 'Fp2','AF8','AF4', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'C2', 'C4',
                 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

    for mat_dir in mat_dirs:
        mat_dict = find_significative_matrices(os.path.join(input_dir, mat_dir))
        for mat_name, matrix in mat_dict.items():
            matrix = reorder_matrix(matrix, channel_order)
            plot_and_save_reorg_matrix(matrix, mat_name, os.path.join(input_dir, mat_dir))

    
if __name__ == '__main__':
    main()