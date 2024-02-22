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

    return input_dir, output_dir

i, o = get_paths()
ROIs_sensor = ['all', 'frontal_l', 'frontal_r', 'parietal_l', 'parietal_r', 'occipital_l', 'occipital_r']
ROIs_source = ['caudalmiddlefrontal-lh', 'caudalmiddlefrontal-rh', 'inferiorparietal-lh', 'inferiorparietal-rh',
               'lateraloccipital-lh', 'lateraloccipital-rh']
conds = ['RESTINGSTATEOPEN', 'RESTINGSTATECLOSE']

run_src = True

for directory in sorted(os.listdir(i)):
    if 'sub-' in directory:
        subject_id = directory.split('-')[1]
        for cond in conds:
            for ROI in ROIs_sensor:
                try:
                    rsf.single_subj_pipeline(subject_id, cond, ROI)
                    print(f'Finished with subject {subject_id} and condition {cond} and ROI {ROI}')
                except:
                    print(f'Error with subject {subject_id} -- most likely no data')
                    continue

if run_src:
    for directory in sorted(os.listdir(i)):
        if 'sub-' in directory:
            subject_id = directory.split('-')[1]
            for cond in conds:
                for ROI in ROIs_source:
                    try:
                        rsf.single_subj_pipeline_src(subject_id, cond, ROI)
                        print(f'Finished with subject {subject_id} and condition {cond} and ROI {ROI}')
                    except:
                        print(f'Error with subject {subject_id} -- most likely no data')
                        continue