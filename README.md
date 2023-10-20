# Master_thesis
Scripts for the analysis of the EEG, eyetracking and behavioral data used for my master's thesis

## Table of content 

- [data](#data)
- [subjects_organization](#subject-organization)
- [file_organization](#file-organization)
- [workflow](#typical-workflow)

# data

We have 3 different tasks.

-  Resting state, eyes open and closed : EEG data

- Singleton visual search task (simplified to N2pc in most files): EEG and behavioral data

- Free viewing task (simplified to Alpheye in most files): EEG, eyetracking and behavioral data

# subjects organization

- The healthy participants are named from sub-01 to sub-24. Excluded participants (bad data quality - too many interpolated electrodes): [8, 9, 10, 11, 15]

- The stroke participants are named with digits above 50 (i.e sub-51). No exclusion yet. 

- We plan on having a group of younger controls, it is not clear yet how they will be named. 

# file organization 

- EEG 
    - N2pc
        - files for analysis
        - function directory
              - func module 1
              - func module 2
              - ...
    - resting-state
        - files for analysis
        - function directory
              - func module 1
              - func module 2
              - ...
    - Alpheye
        - files for analysis
        - function directory
              - func module 1
              - func module 2
              - ...
- Eye-tracking
    - Alpheye
- toBIDS
    - files to get raw -> BIDS
    - data
        - sourcedata
        - derivatives

# typical workflow

1. If not already the case, transform the data to BIDS format. 
2. Preprocess.
3. Analyis for each task.

