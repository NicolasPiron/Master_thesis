import resting_func.static_connectivity as static_connectivity
from resting_func.set_paths import get_paths
from resting_func.set_subject_lists import get_subject_list
import numpy as np

input_dir, output_dir = get_paths()
subject_list = get_subject_list()

pulvinar = ['51', '53', '59', '60']
old_control = [sub for sub in subject_list if int(sub) < 50]
thal_control = ['52', '54', '55', '56', '58']

right_lesion = ['51', '53', '54', '58', '59']
metric = 'ciplv'
freqs = [np.arange(4,9), np.arange(8, 13), np.arange(12, 17), np.arange(16, 31)]

for subject_id in subject_list:
    subject_id = str(subject_id)
    for condition in ['RESTINGSTATEOPEN', 'RESTINGSTATECLOSE']:
        for freq in freqs:
            try:
                if subject_id in right_lesion:
                    static_connectivity.get_conn_src(subject_id, condition, metric=metric,
                                                    freqs=freq, input_dir=input_dir, invert_sides=True)
                else:
                    static_connectivity.get_conn_src(subject_id, condition, metric=metric,
                                                    freqs=freq, input_dir=input_dir, invert_sides=False)
            except:
                print(f'Error in {subject_id} - {condition} - {freq}')
                continue


