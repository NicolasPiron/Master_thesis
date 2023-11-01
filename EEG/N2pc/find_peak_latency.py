from n2pc_func.ERP import get_peak_latency_grand_average
from n2pc_func.set_paths import get_paths
from n2pc_func.set_subject_lists import get_excluded_subjects_list

input_dir, output_dir = get_paths()
excluded_subjects_list = get_excluded_subjects_list()

if __name__ == '__main__':
    get_peak_latency_grand_average(input_dir, output_dir, excluded_subjects_list=excluded_subjects_list)