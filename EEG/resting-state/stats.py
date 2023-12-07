from resting_func.conn_stats import get_nbs_inputs, nbs_bct_corr_z, nbs_report, global_pval_df, plot_bin_mat
from resting_func.static_connectivity import create_significant_conn_mat, plot_significant_conn_mat
from resting_func.set_paths import get_paths
import numpy as np
import pandas as pd
import os
import glob
from itertools import combinations

input_dir, output_dir = get_paths()

def run_nbs():
    '''
    Runs NBS on all combinations of groups, with different thresholds, frequencies and conditions
    '''

    group_dict = {'old_control': [1, 2, 3, 4, 6, 7, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23],
                    'young_control': [70, 71, 72, 73, 74, 75, 76, 77, 78],
                    'thal_control': [52, 54, 55, 56, 58],
                    'pulvinar': [51, 53, 59, 60]
    }
    pairs = list(combinations(group_dict.keys(), 2))
    freqs_dict = {'theta': np.arange(4, 9),
                    'alpha': np.arange(8, 13),
                    'low_beta': np.arange(12, 17),
                    'high_beta': np.arange(16, 31),
    }
    thresh_list = [0.5, 0.6, 0.7]
    condition_list = ['RESTINGSTATEOPEN', 'RESTINGSTATECLOSE']
    metric='plv'

    for thresh in thresh_list:
        for pair in pairs:
            for i, freqs in enumerate(freqs_dict.values()):
                for condition in condition_list:
                    
                    try:
                        list_1 = group_dict[pair[0]]
                        list_2 = group_dict[pair[1]]

                        pop_dict1 = {'subject_list': list_1,
                                        'freqs': freqs,
                                        'metric': metric,
                                        'condition': condition
                        }

                        pop_dict2 = {'subject_list': list_2,
                                        'freqs': freqs,
                                        'metric': metric,
                                        'condition': condition
                        }

                        # Get inputs
                        mat_list, y_vec = get_nbs_inputs(input_dir, pop_dict1, pop_dict2)

                        # Run NBS
                        print(f'Running NBS for {pair[0]} vs {pair[1]} for {freqs} Hz, {condition} condition, threshold {thresh}')
                        pvals, adj, null = nbs_bct_corr_z(mat_list, thresh=thresh, y_vec=y_vec)

                        # Save report
                        def get_prefix(string):
                            return string.split('_')[0]
                        prefix1 = get_prefix(pair[0])
                        prefix2 = get_prefix(pair[1])
                        freq_string = list(freqs_dict.keys())[i]

                        if condition == 'RESTINGSTATEOPEN':
                            name1 = f'{prefix1}-{freq_string}-open'
                            name2 = f'{prefix2}-{freq_string}-open'
                        elif condition == 'RESTINGSTATECLOSE':
                            name1 = f'{prefix1}-{freq_string}-closed'
                            name2 = f'{prefix2}-{freq_string}-closed'

                        nbs_report(pvals, adj, null, thresh, output_dir, name1, name2)
                    except:
                        print(f'Error with {pair[0]} vs {pair[1]} for {freqs} Hz, {condition} condition, threshold {thresh}')
                        continue
    return None

if __name__ == '__main__':
    run_nbs()
    global_pval_df(input_dir, output_dir)
    plot_bin_mat(input_dir)
    create_significant_conn_mat(input_dir, output_dir)
    plot_significant_conn_mat(input_dir, output_dir)

