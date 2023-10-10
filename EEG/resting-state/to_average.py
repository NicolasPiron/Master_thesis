import mne
import os
import glob

# This file computes the average of the epochs for each subject and each condition (eyes closed and eyes open)
#####################################################################
# parameters to be changed by user

# path to the derivative folder 
input_dir = ''

# path to the output folder
output_dir = ''

#####################################################################

def to_average(input_dir, output_dir):

    # list of subjects
    subjects_dirs = glob.glob(os.path.join(input_dir, 'sub-*'))

    # for each condition
    for condition in ['RESTINGSTATEOPEN', 'RESTINGSTATECLOSED']:

        # loop over subjects and create the average
        for dir in subjects_dirs:

            subject = os.path.basename(dir)
            # path to epochs
            epochs_path = glob.glob(os.path.join(dir, condition, 'cleaned_epochs', 'sub-*.fif'))[0]

            # check id the file exists
            if os.path.exists(epochs_path):
                epochs = mne.read_epochs(epochs_path)

                # create the average
                average = epochs.average()

                # save the average
                if not os.path.join(output_dir, subject, condition, 'average'):
                    os.makedirs(os.path.join(output_dir, subject, condition, 'average'))
                average.save(os.path.join(output_dir, subject, condition, 'average', f'{subject}_{condition}_average-epo.fif'))

                print(f'========== Average {condition} computed for subject {subject}')
            else:
                print(f'========== No epochs found for subject {subject}, condition {condition}')

if __name__ == '__main__':
    to_average(input_dir, output_dir)

