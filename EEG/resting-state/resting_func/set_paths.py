import os

def get_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)
    if 'nicolaspiron/Documents' in script_dir:
        input_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'
        output_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'
    elif 'shared_PULSATION' in script_dir:
        input_dir = '/home/nicolasp/shared_PULSATION/derivative'
        output_dir = '/home/nicolasp/shared_PULSATION/derivative'
    elif '/Users/pironn' in script_dir:
        input_dir = '/Users/pironn/Documents/Master/data'
        output_dir = '/Users/pironn/Documents/Master/data'
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

