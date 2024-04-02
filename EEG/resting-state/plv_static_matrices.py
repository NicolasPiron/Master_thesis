from resting_func.set_paths import get_paths
from resting_func.set_subject_lists import get_subject_list
import resting_func.static_connectivity as static_connectivity
import numpy as np

# Define paths
input_dir, output_dir = get_paths()
subject_list = get_subject_list()

pulvinar = ['51', '53', '59', '60']
old_control = [sub for sub in subject_list if sub < 50]
thal_control = ['52', '54', '55', '56', '58']

group_dict = {'old_control':old_control, 'thal_control':thal_control, 'pulvinar':pulvinar}

# Define parameters
subject_list = ['51', '53', '54', '58', '59']
metric_list = ['ciplv']
freqs_list = [np.arange(4,9), np.arange(8, 13), np.arange(12, 17), np.arange(16, 31)]

if __name__ == '__main__':

#     for subject_id in subject_list:
#         if subject_id in ['51', '53', '54', '58', '59']:
#             invert_sides = True
#         else:
#             invert_sides = False
#         for metric in metric_list:
#             for freqs in freqs_list:
#                 try:
#                     print(f'Creating matrix for subject {subject_id}, {metric}, {freqs}')
#                     static_connectivity.create_conn_matrix_subject(subject_id, metric, freqs, input_dir, output_dir, invert_sides=invert_sides)
#                 except:
#                     print(f'Could not create matrix for subject {subject_id}, {metric}, {freqs}')
#                     continue

    for group_name, group_list in group_dict.items():
       group_list = sorted(group_list)
       for metric in metric_list:
           for freqs in freqs_list:
               try:
                   df_open_group, df_closed_group = static_connectivity.create_conn_matrix_group(group_list, metric, freqs, input_dir, output_dir)
                   static_connectivity.plot_and_save_group_matrix(df_open_group, df_closed_group, group_name, metric, freqs, output_dir)
               except:
                   print(f'Could not create matrix for {group_name}, {metric}, {freqs}')
                   continue





