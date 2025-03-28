import os
import resting_func.rs_frequency as rsfr

def get_paths():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if 'nicolaspiron/Documents' in script_dir:
        input_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'
        output_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'
    elif 'shared_PULSATION' in script_dir:
        input_dir = '/home/nicolasp/shared_PULSATION/derivative'
        output_dir = '/home/nicolasp/shared_PULSATION/derivative'

    return input_dir, output_dir

i, o = get_paths()
for directory in sorted(os.listdir(i)):
    if 'sub-' in directory:
        subject_id = directory.split('-')[1]
        try:
            for cond in ['RESTINGSTATEOPEN', 'RESTINGSTATECLOSE']:
                rsfr.save_all_powers(subject_id, cond)
                print(f'Finished with subject {subject_id} and condition {cond}')
        except Exception as e:
            print(e)
            print(f'Error with subject {subject_id} -- most likely no data')
            continue
