import sys
from functions.file_management import concat_all_subj

# Task is either 'N2pc' or 'Alpheye'
# Population is either 'control' or 'stroke'

#input_dir = '/home/nicolasp/shared_PULSATION/derivative'
input_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'

output_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc/all_subj'

#output_dir = '/home/nicolasp/shared_PULSATION/derivative/all_subj'

exclude_subject_list = [9, 10, 15, 20]

if __name__ == '__main__':
    
    # Check that there are 2 arguments (including the script name)
    if len(sys.argv) != 3:
        print("Usage: python concat_allsubj.py <arg1> <arg2>")
        sys.exit(1)

    # Get the arguments from the command line
    task = sys.argv[1]
    population = sys.argv[2]

    # Concatenate the files
    concat_all_subj(task, population, input_dir, output_dir, exclude_subject=False, exclude_subject_list=exclude_subject_list)



