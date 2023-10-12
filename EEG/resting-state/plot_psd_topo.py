from time_frequency_func import plot_resting_psd, plot_resting_spectral_topo
import os
import glob

# This file plots the topography of the scalp for each subject (theta and alpha bands) 
# It also plots the PSD for each cluster of electrodes (occip, frontal, parietal, total)

#####################################################################
# parameters to be changed by user
input_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'
output_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'

bands = {'Theta':(4, 8),
        'Alpha':(8,12)}

cluster_dict = {'occipital': {'right':['O2', 'PO4', 'PO8'], 'left': ['O1', 'PO3', 'PO7']},
                'parietal' : {'right':['P2', 'CP2', 'CP4'], 'left':['P1', 'CP1', 'CP3']},
                'frontal':{'right':['FC2', 'FC4', 'F2'], 'left':['FC1', 'FC3', 'F1']},
                'total':{ 'right':['O2', 'PO4', 'PO8', 'P2', 'CP2', 'CP4', 'FC2', 'FC4', 'F2'],
                            'left':['O1', 'PO3', 'PO7', 'P1', 'CP1', 'CP3', 'FC1', 'FC3', 'F1']}}

#####################################################################


def loop_and_plot(input_dir, output_dir, bands, cluster_dict):

    # list of subjects
    subjects_dirs = glob.glob(os.path.join(input_dir, 'sub-*'))

    # loop over subjects and create the average
    for dir in subjects_dirs:

        # get the subject id
        subject = os.path.basename(dir)
        subject_id = subject.split('-')[1]

        print(f'========== Working on subject {subject_id}')

        try:

            # plot the topography of the scalp for each subject
            plot_resting_spectral_topo(subject_id, input_dir, output_dir, bands)

            # plot the PSD for each cluster of electrodes
            for cluster in cluster_dict.values():

                for side in cluster.keys():

                    plot_resting_psd(subject_id, input_dir, output_dir, picks=cluster[side])
        
        except:

            print(f'========== No data found for subject {subject_id}')


if __name__ == '__main__':
    loop_and_plot(input_dir=input_dir, output_dir=output_dir, bands=bands, cluster_dict=cluster_dict)
