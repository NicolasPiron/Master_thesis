import mne
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def get_rejected_trials_proportion(subject_id, input_dir, output_dir):

    # Load the orginal event list (before any rejection(wrong response, autoreject, heog))
    original_elist = pd.read_csv(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'preprocessing', 'step-03-event_lists', f'sub-{subject_id}-elist-N2pc.csv'))

    # Load the last version of cleaned epochs
    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', 'heog-artifact-rejected', f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))

    # Get the events from the epochs object
    events = epochs.events
    events_df = df = pd.DataFrame(events, columns=['timepoint', 'duration', 'stim'])

    # Get the number of trials for each category in the original event list
    dis_mid_original = (original_elist['stim'] == 1).sum() + (original_elist['stim'] == 2).sum() + (original_elist['stim'] == 5).sum() + (original_elist['stim'] == 6).sum()
    dis_contra_original = (original_elist['stim'] == 7).sum() + (original_elist['stim'] == 8).sum()
    no_dis_original = (original_elist['stim'] == 3).sum() + (original_elist['stim'] == 4).sum()

    # Create counters for each category
    dis_mid_keeped = 0
    dis_contra_keeped = 0
    no_dis_keeped = 0

    # Loop over the original event list and check if the timepoint is in the events_df
    # Then, add 1 to the corresponding counter
    for i, row in original_elist.iterrows():
        t=row['timepoint']
        result = events_df.isin([t]).any().any()
        if result:
            row = np.where(original_elist['timepoint'] == t)
            stimulus_cat = original_elist.loc[row]['stim'].values[0]
            if stimulus_cat in [1, 2, 5, 6]:
                dis_mid_keeped += 1
            elif stimulus_cat in [7, 8]:
                dis_contra_keeped += 1
            elif stimulus_cat in [3, 4]:
                no_dis_keeped += 1

    # Get the number of rejected trials for each category
    dis_mid_rejected = dis_mid_original - dis_mid_keeped
    dis_contra_rejected = dis_contra_original - dis_contra_keeped
    no_dis_rejected = no_dis_original - no_dis_keeped

    # Get the proportion of rejected trials for each category
    dis_mid_rejected_proportion = dis_mid_rejected / dis_mid_original
    dis_contra_rejected_proportion = dis_contra_rejected / dis_contra_original
    no_dis_rejected_proportion = no_dis_rejected / no_dis_original

    if dis_mid_rejected_proportion > 0.25 or dis_contra_rejected_proportion > 0.25 or no_dis_rejected_proportion > 0.25:
        print(f'WARNING: Subject {subject_id} has more than 20% rejected trials in at least one category')
        print(f'dis_mid_rejected_proportion: {dis_mid_rejected_proportion}')
        print(f'dis_contra_rejected_proportion: {dis_contra_rejected_proportion}')
        print(f'no_dis_rejected_proportion: {no_dis_rejected_proportion}')

    # Create a dataframe with the results
    df = pd.DataFrame({'dis_mid_original': [dis_mid_original],
                       'dis_mid_rejected': [dis_mid_rejected],
                       'dis_mid_rejected_proportion': [dis_mid_rejected_proportion],
                       'dis_contra_original': [dis_contra_original],
                       'dis_contra_rejected': [dis_contra_rejected],
                       'dis_contra_rejected_proportion': [dis_contra_rejected_proportion],
                       'no_dis_original': [no_dis_original],
                       'no_dis_rejected': [no_dis_rejected],
                       'no_dis_rejected_proportion': [no_dis_rejected_proportion]})
    
    # Add the subject id to the dataframe at the first position
    df.insert(0, 'ID', subject_id)

    # save the dataframe as a csv file
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'total-trial-rejection')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'total-trial-rejection'))
    df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'total-trial-rejection', f'sub-{subject_id}-trial_rejection.csv'), index=False)
    print(f'================ Dataframe created and saved for subject {subject_id} ================')
 
    return df

def get_rejected_trials_proportion_all_subj(input_dir, output_dir):

    # Create the subject list
    subject_list = [sub for sub in os.listdir(input_dir) if sub.startswith('sub-')]
    subject_list = sorted(subject_list)

    # Create a list to store the dataframes
    df_list = []

    # Loop over the subjects and check if the dataframe exists, otherwise create it
    for subject in subject_list:
        subject_id = subject[-2:]
        if os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'total-trial-rejection', f'sub-{subject_id}-trial_rejection.csv')):
            print(f'================ Dataframe exists for subject {subject_id} ================')
            df = pd.read_csv(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'total-trial-rejection', f'sub-{subject_id}-trial_rejection.csv'))
        else:
            try:
                print(f'================ Dataframe does not exist for subject {subject_id} - creating it ================')
                df = get_rejected_trials_proportion(subject_id, input_dir, output_dir)
            except:
                print(f'================ Impossible to create the dataframe for {subject_id} ================')
                continue
        df_list.append(df)

    # Concatenate all dataframes in the list
    df = pd.concat(df_list)

    # Save dataframe as .csv file
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'total-trial-rejection')):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'total-trial-rejection'))
    df.to_csv(os.path.join(output_dir, 'all_subj', 'N2pc', 'total-trial-rejection', 'trial_rejection-allsubj.csv'), index=False)
    print(f'================ Dataframe created and saved for all subjects! :)')

    # Check if there are subjects with more than 25% rejected trials in at least one category
    proportion_df = df.copy().drop(columns=['dis_mid_original', 'dis_mid_rejected', 'dis_contra_original', 'dis_contra_rejected',
                           'no_dis_original', 'no_dis_rejected'])
    # Rename the columns
    proportion_df = proportion_df.rename(columns={'dis_mid_rejected_proportion': 'Dis mid',
                                                    'dis_contra_rejected_proportion': 'Dis contra',
                                                    'no_dis_rejected_proportion': 'No dis'})

    # Transform the dataframe from wide to long format
    proportion_df = proportion_df.melt(id_vars=['ID'], var_name='condition', value_name='rejection proportion')

    # Create a figure
    fig = sns.catplot(
        data=proportion_df, kind="bar",
        x="ID", y="rejection proportion", hue='condition', palette="dark", alpha=.6, height=6)
    fig.refline(y=0.25, color='red')
    fig.set_xticklabels(rotation = (45), fontsize = 10)
    fig.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'total-trial-rejection', 'trial_rejection-allsubj.png'), dpi=300)

    return None

def plot_rejection_proportion(subject_id, input_dir, output_dir):

    # Check if the dataframe exists
    if os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'total-trial-rejection', f'sub-{subject_id}-trial_rejection.csv')):
        df = pd.read_csv(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'total-trial-rejection', f'sub-{subject_id}-trial_rejection.csv'))
        print('================ Dataframe loaded ================')
    else:
        print('================ Dataframe not found - creating it ================')
        df = get_rejected_trials_proportion(subject_id, input_dir, output_dir)

    # Create a figure

    # Get the proportions of rejected trial for each category in the df
    values = np.array([df.iloc[0,3], df.iloc[0,6], df.iloc[0,9]])
    x = ['Dis mid', 'Dis contra', 'No dis']
    
    # Define a threshold (25%)
    threshold = 0.25
    above_threshold = np.maximum(values - threshold, 0)
    below_threshold = np.minimum(values, threshold)

    fig, ax = plt.subplots()
    ax.bar(x, below_threshold, 0.50, color="lightblue")
    ax.bar(x, above_threshold, 0.50,color="#190482",
            bottom=below_threshold)
    ax.set_ylim(0,1)
    ax.set_ylabel('percentage of rejected trials')
    ax.set_xlabel('conditions')

    ax.axhline(y=0.25,color='#164863', label='25% rejection threshold')
    ax.legend()
    fig.set_size_inches([5,5])
    fig.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'total-trial-rejection', f'sub-{subject_id}-trial_rejection.png'), dpi=300)
    plt.close(fig)

    return None

