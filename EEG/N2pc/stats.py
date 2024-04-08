import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import random
import os
import mne
from mne import io
from mne.stats import bonferroni_correction, fdr_correction
from n2pc_func.set_paths import get_paths


def get_n2pc_array_subject(subject_id):
    '''
    Get the N2pc array for a given subject

    Parameters
    ----------
    subject_id : str
        The subject id
    
    Returns
    -------
    X : np.array
        The N2pc array of shape (n_trials, n_timepoints)
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

    x_left = target_left.get_data(picks=['PO8']) - target_left.get_data(picks=['PO7'])
    x_right = target_right.get_data(picks=['PO7']) - target_right.get_data(picks=['PO8'])
    x = np.concatenate((x_left, x_right), axis=0)
    X = x.reshape(x.shape[0], x.shape[2])
    return X, times

def get_n2pc_array_group(subject_list):

    array_list = []
    for subject_id in subject_list:
        x, times = get_n2pc_array_subject(subject_id)
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

def plot_n2pc(T, times, threshold_fdr, threshold_uncorrected, group):
    '''
    Plot T values of N2pc array
    '''

    _, o = get_paths()
    path = os.path.join(o, 'all_subj', 'N2pc', 'stats', 't-test')
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
    ax.set_title(f"{group} N2pc T-test")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("T-stat")
    plt.tight_layout()
    #plt.show()
    fig.savefig(os.path.join(path, f'{group}-ttest.png'))


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

def permutations(subject_list, n_subjects, k, n_epochs):
    
    all_subjects_data = load_data(subject_list)
    t_values = []
    rejects_fdr = []
    pvals = []
    pvals_fdr = []
    thresholds_fdr = []
    
    for i in range(k):
        sample = random.sample(subject_list, n_subjects)
        data = extract_data(sample, all_subjects_data)
        X, times = concat_data(data)
        #if X.shape[0] > n_epochs:
        #    X = X[0:n_epochs,:]
        #elif X.shape[0] < n_epochs:
        #    print('WARNING - NOT ENOUGH EPOCHS')
        T, pval, reject_fdr, pval_fdr, threshold_fdr, threshold_uncorrected = stats_n2pc(X)
        t_values.append(T)
        rejects_fdr.append(reject_fdr)
        pvals.append(pval)
        pvals_fdr.append(pval_fdr)
        thresholds_fdr.append(threshold_fdr)
         
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
    ax.set_title(f"N2pc T-test {group} - Permutations")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("T-stat")
    plt.tight_layout()
    #plt.show()
    fig.savefig(os.path.join(path, f'{group}-ttest.png'))


def main_permutations():

    group_dict = {'old':['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23'],
                  'thalamus': ['52', '54', '55', '56', '58'],
                  'pulvinar':['51', '53', '59', '60'],
                    'young': ['70', '71', '72', '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87']
                    }
    #group_dict = {'test':['01', '02', '03']}
    for group, subject_list in group_dict.items():
        try:
            t_values,_, _, _, threshold_fdr, times = permutations(subject_list, 4, 2000, 100)
            plot_permutations(t_values, threshold_fdr, times, group)
        except:
            print(f'Error in group {group}')
            continue

def main():

    group_dict = {'old':['18', '19', '20', '21', '22'],
                 'pulvinar':['51', '53', '59', '60'],
                  'thalamus': ['52', '54', '55', '56', '58'],
                   'young': ['71', '72', '75', '76', '77']
                   }

    #group_dict = {'test':['01', '02', '03']}
    
    for group, subject_list in group_dict.items():
        try:
            X, times = get_n2pc_array_group(subject_list)
            T, _, _, _, threshold_fdr, threshold_uncorrected = stats_n2pc(X)
            plot_n2pc(T, times, threshold_fdr, threshold_uncorrected, group)
        except:
            print(f'Error in group {group}')
            continue
  

if __name__ == '__main__':
    main()
    main_permutations()