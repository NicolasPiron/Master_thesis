# N2PC - VISUAL SEARCH TASK

The files in this directory are used to further preprocess and analyse the data of a simple singleton visual search task. 

VS task Conditions (8) : - no distractor, target left or right
                         - midline distractor (top, bottom), target left or right
                         - side distractor, target left or right

# Run the scripts in the right order

( /!\ The preprocessing via preproc_pipeline.py has to be done 1st)

0. Check the set_paths.py and set_subject_lists.py. They need to be updated for the files to find the data (at least set_paths.py)
The subjects and the excluded subjects are specified in set_subject_lists.py

1. To avoid problems : to_evoked.py should always be run 1st. 
2. Then combine_evoked.py

After that it's possible to run all the other ones :

3. n2pc_analysis.py -> plots the n2pc waveforms and the numerical values (diff ipsi-contra). You have to run it from the command line and specify 'single' or 'GA' when you call it. 'single' will run the analysis on every subject individually. 'GA' will compute and plot the grand average - be careful to check set_subject_lists.py. 

4. plot_topography.py follows the same logic and plot topographies for the N2pc component AND spectral topographies

5. plot_heog is used to get an ERP like plot of eye movements for each participants, as well as a list of epochs that contain a saccade and the plots of saccadic movement for each epoch. 

6. compute_alpha_epoch.py gives you the alpha power for sub-sets of electrodes for each epoch. You have to specifiy 'single' or 'all' on the command line. If you specify 'single', you have to add the code of the participant after (i.e. single 01)

# Legacy - to be deleted

alpha_power_by_elec.py is not used anymore (to be updated?)

concat_allsubj.py neither (and makes no sens - the goal was to concatenate all the epochs together ?_?)
