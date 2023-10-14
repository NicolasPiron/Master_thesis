# EEG data analysis

3 tasks 

1. resting-state
2. N2pc - visual search task
3. Alpheye - free viewing task

the 3 can be analysed seprately, but before doing so, they need to be preprocessed. 

# Preprocessing

0. Create a directory 'derivatives' (name not important but it's what we chose following BIDS - this is where the data and plots will be saved). 
The raw data is in 'sourcedata'

1. Add the path at the beginning preproc_pipeline.py

2. Run 'python3 preproc_pipeline.py 01' on the command line to preprocess sub-01. 

3. You will have to select the bad channels manually. For that to be possible, you need to have mne-qt-browser installed. The other step that require manual intervention is the choice of the component(s) to exclude after the ICA was done. 

4. That's it

