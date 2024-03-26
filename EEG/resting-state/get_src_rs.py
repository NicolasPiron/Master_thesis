from resting_func.src_rec_rs import create_stc_epochs
from resting_func.set_paths import get_paths
from resting_func.set_subject_lists import get_subject_list

subject_list = get_subject_list()

for subject_id in subject_list:
    subject_id = str(subject_id)
    create_stc_epochs(subject_id)
