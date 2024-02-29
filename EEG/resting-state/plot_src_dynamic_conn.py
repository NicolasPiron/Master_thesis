import os
import resting_func.dynamic_connectivity as dc

def get_paths():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if 'nicolaspiron/Documents' in script_dir:
        input_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'
        output_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'
    elif 'shared_PULSATION' in script_dir:
        input_dir = '/home/nicolasp/shared_PULSATION/derivative'
        output_dir = '/home/nicolasp/shared_PULSATION/derivative'

    return input_dir, output_dir

band_names = ['theta', 'alpha', 'low_beta', 'high_beta']
band_values = [(4, 8), (8, 12), (13, 16), (16, 30)]
bands = list(zip(band_names, band_values))
conditions = ['RESTINGSTATEOPEN', 'RESTINGSTATECLOSE']

i, o = get_paths()
for directory in sorted(os.listdir(i)):
    if 'sub-' in directory:
        subject_id = directory.split('-')[1]
        try:
            for cond in conditions:
                for band in bands:
                    dc.pipeline_src(subject_id, cond, band, spe_indices=False)
                    print(f'Finished with subject {subject_id} and condition {cond} and band {band}')
        except:
            print(f'Error with subject {subject_id} -- most likely no data')
            continue
