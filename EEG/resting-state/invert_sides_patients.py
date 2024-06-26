from resting_func.invert_sides import invert_sides
from resting_func.set_paths import get_paths

input_dir, output_dir = get_paths()

subject_list = [52, 55, 56, 60]

for subject_id in subject_list:
    for condition in ['RESTINGSTATEOPEN', 'RESTINGSTATECLOSE']:
        invert_sides(subject_id, condition, input_dir)
