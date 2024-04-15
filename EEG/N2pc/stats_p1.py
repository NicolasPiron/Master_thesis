import mne
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from mne.stats import bonferroni_correction, fdr_correction
from n2pc_func.set_paths import get_paths

def get_p1_array_subject(subject_id):
    '''
    Get the P1 array for a given subject

    Parameters
    ----------
    subject_id : str
        The subject id
    
    Returns
    -------
    X : np.array
        The P1 array of shape (n_trials, n_timepoints)
    '''

    i, _ = get_paths()
    path = os.path.join(i, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif')
    epochs = mne.read_epochs(path)
    times = 1e3 * epochs.times
    X = epochs.get_data(picks=['O1', 'O2', 'PO3', 'PO4', 'PO7', 'PO8']).mean(axis=1) 

    return X, times

x, _ = get_p1_array_subject('01')
print(x.shape)

def get_p1_array_group(subject_list):

    array_list = []
    for subject_id in subject_list:
        x, times = get_p1_array_subject(subject_id)
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

def plot_p1(T, times, threshold_fdr, threshold_uncorrected, group):
    '''
    Plot T values of P1 array
    '''

    _, o = get_paths()
    path = os.path.join(o, 'all_subj', 'N2pc', 'stats', 'P1', 't-test')
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

    ymin, ymax = ax.get_ylim()
    ax.set_xlim(-200, 800)
    #ax.vlines(peak_t, ymin, ymax, linestyle="--", colors="gray")
    # add a grey rectangle to highlight the peak
    ax.grid(True)
    ax.fill_between([peak_t - 25, peak_t + 25], ymin, ymax, color="blue", alpha=0.2)
    #ax.set_ylim(-9, 9)
    ax.legend()
    ax.set_title(f"{group} P1 T-test - peak at {peak_t:.0f} ms")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("T-stat")
    plt.tight_layout()
    #plt.show()
    fig.savefig(os.path.join(path, f'{group}-ttest.png'))

def plot_p1_subject(T, times, threshold_fdr, threshold_uncorrected, subject_id):
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

def main():

    group_dict = {'old':['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23'],
                  'thalamus': ['52', '54', '55', '56', '58'],
                  'pulvinar':['51', '53', '59', '60'],
                    'young': ['70', '71', '72', '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87']
                    }

    #group_dict = {'test':['01', '02', '03']}
    
    for group, subject_list in group_dict.items():
        try:
            X, times = get_p1_array_group(subject_list)
            T, _, _, _, threshold_fdr, threshold_uncorrected = stats_p1(X)
            plot_p1(T, times, threshold_fdr, threshold_uncorrected, group)
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

if __name__ == '__main__':
    #main()
    main_single_subject()