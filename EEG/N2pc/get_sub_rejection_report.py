import n2pc_func.subject_rejection as subject_rejection
from n2pc_func.set_paths import get_paths
import os

##############################################################################################################

# Path to data
input_dir, output_dir = get_paths()

##############################################################################################################

subject_list = [sub for sub in os.listdir(input_dir) if sub.startswith('sub-')]
subject_list = sorted(subject_list)

if __name__ == '__main__':

    for subject in subject_list:
        subject_id = subject[-2:]
        subject_rejection.plot_rejection_proportion(subject_id, input_dir, output_dir)

    subject_rejection.get_rejected_trials_proportion_all_subj(input_dir, output_dir)
    