import resting_func.static_connectivity as static_connectivity
from resting_func.set_paths import get_paths
from resting_func.set_subject_lists import get_subject_list

input_dir, output_dir = get_paths()
subject_list = get_subject_list()

right_lesion = [51, 53, 54, 58, 59]
metric = 'ciplv'
freqs = [(4, 8), (8, 12), (12, 16), (16, 30)]

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

