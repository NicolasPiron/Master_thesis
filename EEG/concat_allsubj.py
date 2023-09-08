import sys
from functions.file_management import concat_allsubj

# Task is either 'N2pc' or 'Alpheye'
# Population is either 'control' or 'stroke'

input_dir = 'path/to/derivative'

output_dir = 'path/to/derivative/all_subj'

if __name__ == '__main__':
    
    # Check that there are 2 arguments (including the script name)
    if len(sys.argv) != 3:
        print("Usage: python concat_allsubj.py <arg1> <arg2>")
        sys.exit(1)

    # Get the arguments from the command line
    task = sys.argv[1]
    population = sys.argv[2]

    # Concatenate the files
    concat_allsubj(task, population, input_dir, output_dir)



