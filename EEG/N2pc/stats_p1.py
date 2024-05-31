import mne
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from mne.stats import bonferroni_correction, fdr_correction, permutation_cluster_test
from n2pc_func.set_paths import get_paths

def get_p1_array_subject(subject_id, side=None):
    '''
    Get the P1 array for a given subject

    Parameters
    ----------
    subject_id : str
        The subject id
    side : str
        The side of the target ('left' or 'right'). if None, both sides are considered
    
    Returns
    -------
    X : np.array
        The P1 array of shape (n_trials, n_timepoints)
    '''

    i, _ = get_paths()
    path = os.path.join(i, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif')
    epochs = mne.read_epochs(path)
    times = 1e3 * epochs.times
    if side == 'left':
        X = epochs.get_data(picks=['PO7', 'O1', 'PO3']).mean(axis=1)
    elif side == 'right':
        X = epochs.get_data(picks=['PO8', 'O2', 'PO4']).mean(axis=1)
    else:
        X = epochs.get_data(picks=['O1', 'O2', 'PO3', 'PO4', 'PO7', 'PO8']).mean(axis=1) 

    return X, times

def get_p1_array_group(subject_list, side=None):

    array_list = []
    for subject_id in subject_list:
        x, times = get_p1_array_subject(subject_id, side=side)
        array_list.append(x)
    X = np.concatenate(array_list, axis=0)
    return X, times


def stats_p1(X):
    '''
    Perform statistical tests on P1 array

    Parameters
    ----------
    X : np.array
        The P1 array of shape (n_trials, n_timepoints)

    Returns
    -------
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

def plot_p1(T, times, threshold_fdr, reject_fdr, group, side=None):
    '''
    Plot T values of P1 array
    '''

    _, o = get_paths()
    if side != None:
            path = os.path.join(o, 'all_subj', 'N2pc', 'stats', 'P1', 't-test', 'laterality', 'swapped')
            title = f"{group} P100 T-test - target {side}"
            fname = f'{group}-ttest-{side}.png'
    else:
            path = os.path.join(o, 'all_subj', 'N2pc', 'stats', 'P1', 't-test')
            title = f"{group} P100 One Sample T-test"
            fname = f'{group}-ttest.png'

    if not os.path.exists(path):
        os.makedirs(path)


    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(times, T, color='black')
        
    xmin, xmax = plt.xlim()

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
            print(f"significant time points for {group} P100 component: ")
            for start, end in zip(start_indices, end_indices):
                    try:
                         print(f"from {times[start]:.0f}ms to {times[end]:.0f}ms")
                    except:
                        print('error')

            # Plot each segment with rejections
            for start, end in zip(start_indices, end_indices):

                    plt.plot(times[start:end], T[start:end], color='k', linewidth=2)
                    # if group == 'Young':
                    #     if times[start] >  160 and times[end] < 350:
                    #             plt.fill_between(times[start:end], T[start:end], -threshold_fdr, 
                    #                     where=(T[start:end] < 0) & (T[start:end] < -threshold_fdr), 
                    #                     color='red', alpha=0.3, label='N2pc component')
                    #     if times[start] >  250 and times[end] < 500:
                    #             plt.fill_between(times[start:end], T[start:end], threshold_fdr, 
                    #                     where=(T[start:end] > 0) & (T[start:end] > threshold_fdr), 
                    #                     color='blue', alpha=0.3, label='Pd component')
                    # else:
                    try:
                        if times[start] >  100 and times[end] < 200:
                            print(times[start], times[end])
                            plt.fill_between(times[start:end], T[start:end], threshold_fdr, 
                                    where=(T[start:end] > 0) & (T[start:end] > threshold_fdr), 
                                    color='green', alpha=0.3, label='P100 component')
                    except:
                        print('error')
            if group == 'Healthy' or group == 'Young':                
                plt.legend()
                legend = ax.legend()
                plt.setp(legend.get_texts(), fontsize='12', fontweight='bold')

    plt.xlim(-200, 600)
    plt.grid(color='grey', linewidth=0.5, alpha=0.5)
    plt.xlabel('Time (ms)', fontsize=12, fontweight='bold')
    plt.ylabel('T Values', fontsize=12, fontweight='bold')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(12)
            label.set_fontweight('bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(path, fname), dpi=300)


def plot_p1_subject(T, times, threshold_fdr, threshold_uncorrected, subject_id, save_peak=False):
    '''
    Plot T values of P1 array
    '''

    _, o = get_paths()
    path = os.path.join(o, 'all_subj', 'N2pc', 'stats', 'P1', 't-test', 'individual', f'sub-{subject_id}')
    if not os.path.exists(path):
        os.makedirs(path)

     # find the peak index (max T values between 0 and 200 ms)
    window = np.logical_and(times >= 80, times <= 180)
    peak_t = times[window][np.argmax(T[window])]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, T, "k", label="T-stat")
    xmin, xmax = plt.xlim()
    plot_hline(ax, threshold_uncorrected, xmin, xmax, "r", "p=0.05 (uncorrected)")
    plot_hline(ax, -threshold_uncorrected, xmin, xmax, "r")

    if threshold_fdr != 0:
        plot_hline(ax, threshold_fdr, xmin, xmax, "b", "p=0.05 (FDR)")
        plot_hline(ax, -threshold_fdr, xmin, xmax, "b")

    ax.legend()
    ax.grid(True)
    ymin, ymax = ax.get_ylim()
    ax.fill_between([peak_t - 25, peak_t + 25], ymin, ymax, color="blue", alpha=0.2)
    ax.set_title(f"sub-{subject_id} P1 T-test - peak at {peak_t:.0f} ms")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("T-stat")
    plt.tight_layout()
    #plt.show()
    fig.savefig(os.path.join(path, f'sub-{subject_id}-ttest.png'))

    if save_peak:
        df = pd.DataFrame(columns=['ID', 'peak_latency'])
        df['ID'] = np.zeros(1)
        df.iloc[0, 0] = subject_id
        df.iloc[0, 1] = peak_t
        file_path = os.path.join(o, f'sub-{subject_id}', 'N2pc', 'peak-latency', f'sub-{subject_id}-ttest-P1-peak-latency.csv')
        df.to_csv(file_path)


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
    path = os.path.join(o, 'all_subj', 'N2pc', 'stats', 'P1', 'ANOVA')
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
    ax.set_title("P1 ANOVA")
    ax.legend((h,), ("cluster p-value < 0.05",))
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("f-values")
    plt.tight_layout()
    fig.savefig(os.path.join(path, 'anova.png'))

def plot_pairwise(T_obs, clusters, cluster_p_values, times, groups_dict):

    _, o = get_paths()
    path = os.path.join(o, 'all_subj', 'N2pc', 'stats', 'P1', 'ANOVA')
    if not os.path.exists(path):
        os.makedirs(path)

    g1_values = list(groups_dict.values())[0]
    g2_values = list(groups_dict.values())[1]
    g1_name = list(groups_dict.keys())[0]
    g2_name = list(groups_dict.keys())[1]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4))
    ax.set_title(f'P1 Pairwise F-test - {g1_name} vs {g2_name}')
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

def main():

    # group_dict = {'old':['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23'],
    #               'thalamus': ['52', '54', '55', '56', '58'],
    #               'pulvinar':['51', '53', '59', '60'],
    #                 'young': ['70', '71', '72', '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87']
    #                 }

    group_dict = {'test':['01', '02', '03']}
    
    for group, subject_list in group_dict.items():
        try:
            X, times = get_p1_array_group(subject_list)
            T, pval, reject_fdr, pval_fdr, threshold_fdr, threshold_uncorrected = stats_p1(X)
            plot_p1(T, times, threshold_fdr, reject_fdr, group, side=None)
        except:
            print(f'Error in group {group}')
            continue

def main_sides():

    group_dict = {'old':['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23'],
                  'thalamus': ['52', '54', '55', '56', '58'],
                  'pulvinar':['51', '53', '59', '60'],
                    'young': ['70', '71', '72', '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87']
                    }

    # group_dict = {'test':['01', '02', '03']}
    
    for group, subject_list in group_dict.items():
        for side in ['left', 'right']:
            try:
                X, times = get_p1_array_group(subject_list, side=side)
                T, _, _, _, threshold_fdr, threshold_uncorrected = stats_p1(X)
                plot_p1(T, times, threshold_fdr, threshold_uncorrected, group, side=side)
            except:
                print(f'Error in group {group}')
                continue

def main_single_subject():

    subject_list = ['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23',
                    '51', '52', '53', '54', '55', '56', '58', '59', '60',
                    '70', '71', '72', '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87']

    for subject_id in subject_list:

        X, times = get_p1_array_subject(subject_id)
        T, _, _, _, threshold_fdr, threshold_uncorrected = stats_p1(X)
        plot_p1_subject(T, times, threshold_fdr, threshold_uncorrected, subject_id)

def main_anova():
     
    group_dict = {'old':['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23'],
                  'thalamus': ['52', '54', '55', '56', '58'],
                  'pulvinar':['51', '53', '59', '60']
                    }
    # group_dict = {'old':['01'],
    #              'thalamus':['02'],
    #              'pulvinar':['03']}
  
    old_vals, _ = get_p1_array_group(group_dict['old'])
    thalamus_vals, _ = get_p1_array_group(group_dict['thalamus'])
    pulvinar_vals, times = get_p1_array_group(group_dict['pulvinar'])

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


if __name__ == '__main__':
    main()
    #main_single_subject()
    #main_anova()
    #main_sides()