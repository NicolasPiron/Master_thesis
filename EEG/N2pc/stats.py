import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import mne
from mne import io
from mne.stats import bonferroni_correction, fdr_correction, permutation_cluster_test
from n2pc_func.set_paths import get_paths


def get_n2pc_array_subject(subject_id, side=None, invert=False):
    '''
    Get the N2pc array for a given subject

    Parameters
    ----------
    subject_id : str
        The subject id
    side : str
        The side of the target ('left' or 'right'). if None, both sides are considered
    invert : bool
        If True, left and right sides are inverted
    
    Returns
    -------
    X : np.array
        The N2pc array of shape (n_trials, n_timepoints)
    times : np.array
        The timepoints of the N2pc array
    '''

    i, _ = get_paths()
    path = os.path.join(i, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif')
    epochs = mne.read_epochs(path)
    times = 1e3 * epochs.times

    target_left = mne.concatenate_epochs([epochs['dis_top/target_l'],
                                      epochs['no_dis/target_l'],
                                      epochs['dis_bot/target_l'],
                                      epochs['dis_right/target_l']])
    target_right = mne.concatenate_epochs([epochs['dis_top/target_r'],
                                      epochs['no_dis/target_r'],
                                      epochs['dis_bot/target_r'],
                                      epochs['dis_left/target_r']])
    
    # pretty ugly way to invert the data (-> do like the lesion is on the left)
    if invert:
        target_left, target_right = target_right, target_left
        x_left = target_left.get_data(picks=['PO7']) - target_left.get_data(picks=['PO8'])
        x_right = target_right.get_data(picks=['PO8']) - target_right.get_data(picks=['PO7'])
    else:
        x_left = target_left.get_data(picks=['PO8']) - target_left.get_data(picks=['PO7'])
        x_right = target_right.get_data(picks=['PO7']) - target_right.get_data(picks=['PO8'])

    if side == 'left':
        x = x_left
    elif side == 'right':
        x = x_right
    else:
        x = np.concatenate((x_left, x_right), axis=0)
    X = x.reshape(x.shape[0], x.shape[2])
    return X, times

def get_n2pc_array_group(subject_list, side=None):

    invert_subjects = ['52', '55', '56', '60']
    array_list = []
    for subject_id in subject_list:
        if subject_id in invert_subjects:
            x, times = get_n2pc_array_subject(subject_id, side=side, invert=True)
        else:
            x, times = get_n2pc_array_subject(subject_id, side=side)
        array_list.append(x)
    X = np.concatenate(array_list, axis=0)
    return X, times


def stats_n2pc(X):
    '''
    Perform statistical tests on N2pc array

    Parameters
    ----------
    X : np.array
        The N2pc array of shape (n_trials, n_timepoints)

    Returns
    -------
    T : np.array
        The T-statistics of the N2pc array
    pval : np.array
        The p-values of the T-statistics
    reject_fdr : np.array
        Boolean array indicating the rejected null hypothesis for FDR correction
    pval_fdr : np.array
        The p-values of the FDR corrected T-statistics
    threshold_fdr : float
        The threshold for FDR correction
    threshold_uncorrected : float
        The threshold for uncorrected p-values
    '''
    T, pval = stats.ttest_1samp(X, 0)
    alpha = 0.05

    n_samples, n_tests = X.shape
    threshold_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)

    reject_bonferroni, pval_bonferroni = bonferroni_correction(pval, alpha=alpha)
    threshold_bonferroni = stats.t.ppf(1.0 - alpha / n_tests, n_samples - 1)

    reject_fdr, pval_fdr = fdr_correction(pval, alpha=alpha, method="indep")

    if np.sum(reject_fdr) == 0:
        threshold_fdr = 0
    else:
        threshold_fdr = np.min(np.abs(T)[reject_fdr])

    return T, pval, reject_fdr, pval_fdr, threshold_fdr, threshold_uncorrected


def plot_hline(ax, y, xmin, xmax, color, label=None):
        
        if label:
            ax.hlines(y,xmin,xmax,linestyle="--",colors=color,label=label,linewidth=2)
        else:
            ax.hlines(y,xmin,xmax,linestyle="--",colors=color,linewidth=2)

def plot_n2pc(T, times, threshold_fdr, reject_fdr, group, side=None):
    '''
    Plot T values of N2pc array
    '''

    window = np.logical_and(times >= 180, times <= 400)
    peak_t = times[window][np.argmin(T[window])] # will be used after, if n2pc component is found

    _, o = get_paths()
    if side != None:
            path = os.path.join(o, 'all_subj', 'N2pc', 'stats', 't-test', 'laterality', 'swapped')
            title = f"{group} N2pc T-test - target {side}"
            fname = f'{group}-ttest-{side}.png'
    else:
            path = os.path.join(o, 'all_subj', 'N2pc', 'stats', 't-test')
            title = f"{group} N2pc One Sample T-test"
            fname = f'{group}-ttest.png'
        
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(times, T, color='black')
        
    xmin, xmax = plt.xlim()
    # plot_hline(ax, threshold_uncorrected, xmin, xmax, "r", "p=0.05 (uncorrected)")
    # plot_hline(ax, -threshold_uncorrected, xmin, xmax, "r")
    if threshold_fdr != 0:
            plot_hline(ax, threshold_fdr, xmin, xmax, "red", "p=0.05 (FDR)")
            plot_hline(ax, -threshold_fdr, xmin, xmax, "red")
            changes = np.diff(reject_fdr.astype(int))
            start_indices = np.where(changes == 1)[0] + 1
            end_indices = np.where(changes == -1)[0] + 1

            if reject_fdr[0]:  # Start from the first element if it's True
                    start_indices = np.insert(start_indices, 0, 0)
            if reject_fdr[-1]:  # End at the last element if it ends True
                    end_indices = np.append(end_indices, len(reject_fdr))

            #print the time points when the t value is significant
            print(f"significant time points for {group} N2pc component: ")
            for start, end in zip(start_indices, end_indices):
                    print(f"from {times[start]:.0f}ms to {times[end]:.0f}ms")


            # Plot each segment with rejections
            for start, end in zip(start_indices, end_indices):
                    plt.plot(times[start:end], T[start:end], color='k', linewidth=2)
                    if group == 'Young':
                        if times[start] >  160 and times[end] < 350:
                                plt.fill_between(times[start:end], T[start:end], -threshold_fdr, 
                                        where=(T[start:end] < 0) & (T[start:end] < -threshold_fdr), 
                                        color='red', alpha=0.3, label='N2pc component')
                        if times[start] >  250 and times[end] < 500:
                                plt.fill_between(times[start:end], T[start:end], threshold_fdr, 
                                        where=(T[start:end] > 0) & (T[start:end] > threshold_fdr), 
                                        color='blue', alpha=0.3, label='Pd component')
                    else:
                        if times[start] >  245 and times[end] < 410:
                                plt.fill_between(times[start:end], T[start:end], -threshold_fdr, 
                                        where=(T[start:end] < 0) & (T[start:end] < -threshold_fdr), 
                                        color='red', alpha=0.3, label='N2pc component')
            if group == 'Healthy' or group == 'Young':                
                plt.legend()
                legend = ax.legend()
                plt.setp(legend.get_texts(), fontsize='12', fontweight='bold')

    plt.xlim(-200, 600)
    if group == 'Young':
        plt.ylim(-9, 15)
    else:
        plt.ylim(-9, 4)
    plt.grid(color='grey', linewidth=0.5, alpha=0.5)
    plt.xlabel('Time (ms)', fontsize=12, fontweight='bold')
    plt.ylabel('T Values', fontsize=12, fontweight='bold')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(12)
            label.set_fontweight('bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(path, fname), dpi=300)

def plot_n2pc_subject(T, times, threshold_fdr, threshold_uncorrected, subject_id):
    '''
    Plot T values of N2pc array
    '''

    _, o = get_paths()
    path = os.path.join(o, 'all_subj', 'N2pc', 'stats', 't-test', 'individual', f'sub-{subject_id}')
    if not os.path.exists(path):
        os.makedirs(path)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, T, "k", label="T-stat")
    xmin, xmax = plt.xlim()
    plot_hline(ax, threshold_uncorrected, xmin, xmax, "r", "p=0.05 (uncorrected)")
    plot_hline(ax, -threshold_uncorrected, xmin, xmax, "r")

    if threshold_fdr != 0:
        plot_hline(ax, threshold_fdr, xmin, xmax, "b", "p=0.05 (FDR)")
        plot_hline(ax, -threshold_fdr, xmin, xmax, "b")

    ax.legend()
    ax.set_title(f"sub-{subject_id} N2pc T-test")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("T-stat")
    plt.tight_layout()
    #plt.show()
    fig.savefig(os.path.join(path, f'sub-{subject_id}-ttest.png'))


############################################################################################################
# ANOVAs and pairwise t-tests
############################################################################################################

def run_perm_cluster_test(*groups_vals):

    threshold = None
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
    groups_vals,
    n_permutations=1000,
    threshold=threshold,
    tail=0,
    n_jobs=None,
    out_type="mask",
)
    return T_obs, clusters, cluster_p_values, H0

def plot_anova(T_obs, clusters, cluster_p_values, times):

    _, o = get_paths()
    path = os.path.join(o, 'all_subj', 'N2pc', 'stats', 'ANOVA')
    if not os.path.exists(path):
        os.makedirs(path)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_p_values[i_c] <= 0.05:
            h = ax.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
        else:
            ax.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)

    ax.plot(times, T_obs, "g")
    ax.set_title("N2pc ANOVA")
    ax.legend((h,), ("cluster p-value < 0.05",))
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("f-values")
    plt.tight_layout()
    fig.savefig(os.path.join(path, 'anova.png'))

def plot_pairwise(T_obs, clusters, cluster_p_values, times, groups_dict):

    _, o = get_paths()
    path = os.path.join(o, 'all_subj', 'N2pc', 'stats', 'ANOVA')
    if not os.path.exists(path):
        os.makedirs(path)

    g1_values = list(groups_dict.values())[0]
    g2_values = list(groups_dict.values())[1]
    g1_name = list(groups_dict.keys())[0]
    g2_name = list(groups_dict.keys())[1]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4))
    ax.set_title(f'N2pc Pairwise T-test - {g1_name} vs {g2_name}')
    ax.plot(
        times,
        g1_values.mean(axis=0) - g2_values.mean(axis=0),
        label=f"ERP Contrast ({g1_name} - {g2_name})",
    )
    ax.set_ylabel("electric potential (uV)")
    ax.legend()

    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_p_values[i_c] <= 0.05:
            h = ax2.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
        else:
            ax2.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)

    hf = plt.plot(times, T_obs, "g")
    ax2.legend((h,), ("cluster p-value < 0.05",))
    ax2.set_xlabel("time (ms)")
    ax2.set_ylabel("f-values")
    plt.tight_layout()
    fig.savefig(os.path.join(path, f'pairwise-{g1_name}-vs-{g2_name}.png'))

############################################################################################################
# Main functions
############################################################################################################

def main_single_subject():

    subject_list = ['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23',
                    '51', '52', '53', '54', '55', '56', '58', '59', '60',
                    '70', '71', '72', '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87']

    for subject_id in subject_list:

        X, times = get_n2pc_array_subject(subject_id)
        T, pval, reject_fdr, pval_fdr, threshold_fdr, threshold_uncorrected = stats_n2pc(X)
        plot_n2pc_subject(T, times, threshold_fdr, threshold_uncorrected, subject_id)

def main():

    group_dict = {'Healthy':['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23'],
                  'Thalamus': ['52', '54', '55', '56', '58'],
                  'Pulvinar':['51', '53', '59', '60'],
                    'Young': ['70', '71', '72', '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87']
                    }

    # group_dict = {'test':['01', '02']}
    
    for group, subject_list in group_dict.items():

        try:
            X, times = get_n2pc_array_group(subject_list, side=None)
            print(f'number of trials for group {group}: {X.shape[0]}')
            T, pval, reject_fdr, pval_fdr, threshold_fdr, threshold_uncorrected = stats_n2pc(X)
            plot_n2pc(T, times, threshold_fdr, reject_fdr, group, side=None)
            print(f'n2pc component found in group {group} at {times[np.argmin(T)]}ms')
            print(f't value at peak: {np.min(T)}, p value: {pval[np.argmin(T)]}')
        except:
            print(f'Error in group {group}')
            continue

def main_sides():

    group_dict = {'Healthy':['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23'],
                  'thalamus': ['52', '54', '55', '56', '58'],
                  'pulvinar':['51', '53', '59', '60'],
                    'young': ['70', '71', '72', '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87']
                    }

    # group_dict = {'test':['01', '02', '03']}
    
    for group, subject_list in group_dict.items():
        for side in ['left', 'right']:
            try:
                X, times = get_n2pc_array_group(subject_list, side=side)
                T, _, _, _, threshold_fdr, threshold_uncorrected = stats_n2pc(X)
                plot_n2pc(T, times, threshold_fdr, threshold_uncorrected, group, side=side)
            except:
                print(f'Error in group {group}')
                continue

def main_anova():
     
    group_dict = {'old':['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23'],
                  'thalamus': ['52', '54', '55', '56', '58'],
                  'pulvinar':['51', '53', '59', '60']
                    }
    # group_dict = {'old':['01'],
    #              'thalamus':['02'],
    #              'pulvinar':['03']}
  
    old_vals, _ = get_n2pc_array_group(group_dict['old'])
    thalamus_vals, _ = get_n2pc_array_group(group_dict['thalamus'])
    pulvinar_vals, times = get_n2pc_array_group(group_dict['pulvinar'])

    T_obs, clusters, cluster_p_values, H0 = run_perm_cluster_test(old_vals, thalamus_vals, pulvinar_vals)
    print('H0: ', H0)
    plot_anova(T_obs, clusters, cluster_p_values, times)

    group_dict1 = {'old': old_vals, 'thalamus': thalamus_vals}
    group_dict2 = {'old': old_vals, 'pulvinar': pulvinar_vals}
    group_dict3 = {'thalamus': thalamus_vals, 'pulvinar': pulvinar_vals}

    T_obs1, clusters1, cluster_p_values1, H01 = run_perm_cluster_test(old_vals, thalamus_vals)
    T_obs2, clusters2, cluster_p_values2, H02 = run_perm_cluster_test(old_vals, pulvinar_vals)
    T_obs3, clusters3, cluster_p_values3, H03 = run_perm_cluster_test(thalamus_vals, pulvinar_vals)

    plot_pairwise(T_obs1, clusters1, cluster_p_values1, times, group_dict1) 
    plot_pairwise(T_obs2, clusters2, cluster_p_values2, times, group_dict2)
    plot_pairwise(T_obs3, clusters3, cluster_p_values3, times, group_dict3)

#if __name__ == '__main__':
    #main()
    #main_sides()
    #main_single_subject()
    #main_anova()

############################################################################################################
# unused functions
############################################################################################################

def load_data(subject_list):
    
    subject_data = {}
    for subject in subject_list:
        subject_data[subject] = get_n2pc_array_subject(subject)
    
    return subject_data

def extract_data(subject_list, subject_data):
    
    subset = {}
    for subject_id in subject_list:
        subset[subject_id] = subject_data[subject_id]
        
    return subset

def concat_data(subject_data):
    
    array_list = []
    for data in subject_data.values():
        x = data[0]
        times = data[1]
        array_list.append(x)
    X = np.concatenate(array_list, axis=0)
    return X, times

def resample(subject_list, k=10000, n_epochs=1655):
    
    all_subjects_data = load_data(subject_list)
    t_values = []
    rejects_fdr = []
    pvals = []
    pvals_fdr = []
    thresholds_fdr = []
    
    for _ in range(k):
        data = extract_data(subject_list, all_subjects_data)
        X, times = concat_data(data)
        rand_idx = np.random.choice(X.shape[0], n_epochs, replace=False)
        X_sub = X[rand_idx,:]
        T, pval, reject_fdr, pval_fdr, threshold_fdr, threshold_uncorrected = stats_n2pc(X_sub)
        t_values.append(T)
        rejects_fdr.append(reject_fdr)
        pvals.append(pval)
        pvals_fdr.append(pval_fdr)
        thresholds_fdr.append(threshold_fdr)
        
    print(f'Resampling done for {k} iterations')
    t_values = np.array(t_values)
    rejects_fdr = np.array(rejects_fdr)
    pvals = np.array(pvals)
    pvals_fdr = np.array(pvals_fdr)
    thresholds_fdr = np.array(thresholds_fdr)
        
    return t_values, pvals, reject_fdr, pvals_fdr, thresholds_fdr, times

def plot_permutations(t_values, thresholds_fdr, times, group):

    _, o = get_paths()
    path = os.path.join(o, 'all_subj', 'N2pc', 'stats', 'permutations')
    if not os.path.exists(path):
        os.makedirs(path)

    #m_pvals = pvals.mean(axis=0)
    #m_reject_fdr, m_pval_fdr = fdr_correction(m_pvals, alpha=0.05, method="indep")

    m_thresh_fdr = np.where(thresholds_fdr == 0, np.nan, thresholds_fdr)
    # Compute mean ignoring NaN values
    m_thresh_fdr = np.nanmean(m_thresh_fdr, axis=0)

    m_t_values = np.mean(t_values, axis=0)
    sd_t_values = np.std(t_values, axis=0)
    conf_interval = np.percentile(t_values, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, m_t_values, "k", label="T-stat")
    #ax.fill_between(times, m_t_values - sd_t_values, m_t_values + sd_t_values, color="k", alpha=0.2)
    ax.fill_between(times, conf_interval[0], conf_interval[1], color="k", alpha=0.2)
    xmin, xmax = plt.xlim()

    if np.sum(m_thresh_fdr) != 0:
        plot_hline(ax, m_thresh_fdr, xmin, xmax, "b", "p=0.05 (FDR)")
        plot_hline(ax, -m_thresh_fdr, xmin, xmax, "b")
    ax.legend()
    ax.set_title(f"N2pc T-test {group} - Resampled 1655 trials")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("T-stat")
    plt.tight_layout()
    #plt.show()
    fig.savefig(os.path.join(path, f'{group}-ttest.png'))

def main_resamp():

    group_dict = {'old':['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23'],
                  'thalamus': ['52', '54', '55', '56', '58'],
                    'young': ['70', '71', '72', '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87']
                    }
    # group_dict = {'test':['01', '02', '03']}
    for group, subject_list in group_dict.items():
        try:
            t_values,_, _, _, threshold_fdr, times = resample(subject_list)
            plot_permutations(t_values, threshold_fdr, times, group)
        except:
            print(f'Error in group {group}')
            continue


if __name__ == '__main__':
    main_resamp()