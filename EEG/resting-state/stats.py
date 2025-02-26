from resting_func.conn_stats import get_nbs_inputs, nbs_bct_corr_z, nbs_report, global_pval_df, plot_bin_mat
#from resting_func.static_connectivity import create_significant_conn_mat, plot_significant_conn_mat
from resting_func.set_paths import get_paths
import numpy as np
from itertools import combinations

input_dir, output_dir = get_paths()

def run_pairwise_nbs():

    group_dict = {
        'old_control': [1, 2, 3, 4, 6, 7, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23],
        'thal_control': [52, 54, 55, 56, 58],
        'pulvinar': [51, 53, 59, 60]
    }
    
    pairs = list(combinations(group_dict.keys(), 2))
    freqs_dict = {
        'alpha': np.arange(8, 13), 
        'theta': np.arange(4, 9),
        'low_beta': np.arange(12, 17),
    }
    condition_list = ['RESTINGSTATEOPEN','RESTINGSTATECLOSE']
    metrics=['ciplv']
    metric=metrics[0]

    for pair in pairs:
        for i, freqs in enumerate(freqs_dict.values()):
            for condition in condition_list:
                
                try:
                    list_1 = group_dict[pair[0]]
                    list_2 = group_dict[pair[1]]

                    pop_dict1 = {
                        'subject_list': list_1,
                        'freqs': freqs,
                        'metric': metric,
                        'condition': condition
                    }

                    pop_dict2 = {
                        'subject_list': list_2,
                        'freqs': freqs,
                        'metric': metric,
                        'condition': condition
                    }


                    mat_list, y_vec = get_nbs_inputs(input_dir, pop_dict1, pop_dict2, source=False)

                    # Run NBS
                    print(f'Running NBS for {pair[0]} vs {pair[1]} for {freqs} Hz, {condition} condition - metric {metric}')
                    pvals, adj, null = nbs_bct_corr_z(mat_list, thresh=0.7, y_vec=y_vec)

                    # Save report
                    def get_prefix(string):
                        return string.split('_')[0]
                    prefix1 = get_prefix(pair[0])
                    prefix2 = get_prefix(pair[1])
                    freq_string = list(freqs_dict.keys())[i]

                    if condition == 'RESTINGSTATEOPEN':
                        name1 = f'{prefix1}-{freq_string}-{metric}-open'
                        name2 = f'{prefix2}-{freq_string}-{metric}-open'
                    elif condition == 'RESTINGSTATECLOSE':
                        name1 = f'{prefix1}-{freq_string}-{metric}-closed'
                        name2 = f'{prefix2}-{freq_string}-{metric}-closed'
                    nbs_report(pvals, adj, null, 0.7, output_dir, name1, name2, source=False)
                except:
                    print(f'Error with {pair[0]} vs {pair[1]} for {freqs} Hz, {condition} condition, - metric {metric}')
                    continue

    return None

# {'theta': np.arange(4, 9),
#                     'alpha': np.arange(8, 13),
#                     'low_beta': np.arange(12, 17),
#     }

def run_anovas():
    '''
    Runs ANOVA on all groups, with different metrics, frequencies and conditions
    '''

    group_dict = {'old_control': [1, 2, 3, 4, 6, 7, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23],
                    'thal_control': [52, 54, 55, 56, 58],
                    'pulvinar': [51, 53, 59, 60]
    }

    freqs_dict = freqs_dict = {'theta': np.arange(4, 9),
                    'alpha': np.arange(8, 13),
                    'low_beta': np.arange(12, 17),
                    'high_beta': np.arange(16, 31),
    }
    thresh = 0.7
    cond = ['RESTINGSTATEOPEN', 'RESTINGSTATECLOSE']
    metrics = ['ciplv']

    for metric in metrics:
        for condition in cond:
            for freq_name, freqs in freqs_dict.items():

                try:
                    print(f'Running NBS for {freqs}Hz and {condition} condition, with metric {metric}')
                    old = {'subject_list': group_dict['old_control'],
                                'freqs': freqs,
                                'metric': metric,
                                'condition': condition
                    }
                    thal = {'subject_list': group_dict['thal_control'],
                                'freqs': freqs,
                                'metric': metric,
                                'condition': condition
                    }
                    pulv = {'subject_list': group_dict['pulvinar'],
                                'freqs': freqs,
                                'metric': metric,
                                'condition': condition
                    }

                    name = f'{freq_name}-{condition}-{metric}-ANOVA'

                    mat_list, y_vec = get_nbs_inputs(input_dir, old, thal, pulv, source=False)
                    pvals, adj, null = nbs_bct_corr_z(mat_list, thresh=thresh, y_vec=y_vec)
                    nbs_report(pvals, adj, null, thresh, output_dir, name, source=False)
                except:
                    print(f'Error with {freqs} Hz, {condition} condition, threshold {thresh}')
                    continue



if __name__ == '__main__':
    #run_nbs()
    #plot_bin_mat(input_dir)
    #create_significant_conn_mat(input_dir, output_dir)
    #plot_significant_conn_mat(input_dir, output_dir)

    run_anovas()
    run_pairwise_nbs()

    global_pval_df(input_dir, output_dir, source=False)


############################################################################################################
# unused code

#def run_nbs():
#     '''
#     Runs NBS on all combinations of groups, with different thresholds, frequencies and conditions
#     '''

#     group_dict = {'old_control': [1, 2, 3, 4, 6, 7, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23],
#                     'young_control': [70, 71, 72, 73, 74, 75, 76, 77, 78],
#                     'thal_control': [52, 54, 55, 56, 58],
#                     'pulvinar': [51, 53, 59, 60]
#     }
#     pairs = list(combinations(group_dict.keys(), 2))
#     freqs_dict = {'theta': np.arange(4, 9),
#                     'alpha': np.arange(8, 13),
#                     'low_beta': np.arange(12, 17),
#                     'high_beta': np.arange(16, 31),
#     }
#     thresh_list = [0.5, 0.6, 0.7]
#     condition_list = ['RESTINGSTATEOPEN', 'RESTINGSTATECLOSE']
#     metric='plv'

#     for thresh in thresh_list:
#         for pair in pairs:
#             for i, freqs in enumerate(freqs_dict.values()):
#                 for condition in condition_list:
                    
#                     try:
#                         list_1 = group_dict[pair[0]]
#                         list_2 = group_dict[pair[1]]

#                         pop_dict1 = {'subject_list': list_1,
#                                         'freqs': freqs,
#                                         'metric': metric,
#                                         'condition': condition
#                         }

#                         pop_dict2 = {'subject_list': list_2,
#                                         'freqs': freqs,
#                                         'metric': metric,
#                                         'condition': condition
#                         }

#                         # Get inputs
#                         mat_list, y_vec = get_nbs_inputs(input_dir, pop_dict1, pop_dict2, source=False)

#                         # Run NBS
#                         print(f'Running NBS for {pair[0]} vs {pair[1]} for {freqs} Hz, {condition} condition, threshold {thresh}')
#                         pvals, adj, null = nbs_bct_corr_z(mat_list, thresh=thresh, y_vec=y_vec)

#                         # Save report
#                         def get_prefix(string):
#                             return string.split('_')[0]
#                         prefix1 = get_prefix(pair[0])
#                         prefix2 = get_prefix(pair[1])
#                         freq_string = list(freqs_dict.keys())[i]

#                         if condition == 'RESTINGSTATEOPEN':
#                             name1 = f'{prefix1}-{freq_string}-open'
#                             name2 = f'{prefix2}-{freq_string}-open'
#                         elif condition == 'RESTINGSTATECLOSE':
#                             name1 = f'{prefix1}-{freq_string}-closed'
#                             name2 = f'{prefix2}-{freq_string}-closed'

#                         nbs_report(pvals, adj, null, thresh, output_dir, name1, name2, source=False)
#                     except:
#                         print(f'Error with {pair[0]} vs {pair[1]} for {freqs} Hz, {condition} condition, threshold {thresh}')
#                         continue
#     return None