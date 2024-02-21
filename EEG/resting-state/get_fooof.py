import os
import resting_func.rs_fooof as rsf

def get_paths():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if 'nicolaspiron/Documents' in script_dir:

        input_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'
        output_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'
    
    elif 'shared_PULSATION' in script_dir:

        input_dir = '/home/nicolasp/shared_PULSATION/derivative'
        output_dir = '/home/nicolasp/shared_PULSATION/derivative'

    else:

        print('===================================')
        print('===================================')
        print('WARNING: Running on unknown machine')
        print('===================================')
        print('===================================')
        print('Please set the paths manually')
        input_dir = ''
        output_dir = ''

    return input_dir, output_dir

i, o = get_paths()
ROIs = ['frontal_l', 'frontal_r', 'parietal_l', 'parietal_r', 'occipital_l', 'occipital_r']
conds = ['RESTINGSTATEOPEN', 'RESTINGSTATECLOSE']

for directory in os.listdir(i):
    if 'sub' in directory:
        subject_id = directory.split('-')[1]
        try:
            for cond in conds:
                for ROI in ROIs:
                    rsf.pipeline(subject_id, cond, ROI)
        except:
            print(f'Error with subject {subject_id} -- most likely no data')
            continue

