import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
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


    return T, reject_fdr, pval_fdr, threshold_fdr


def plot_n2pc(T, times, reject_fdr, pval_fdr, threshold_fdr, group):
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

    ax.hlines(
        threshold_fdr,
        xmin,
        xmax,
        linestyle="--",
        colors="b",
        label="p=0.05 (FDR)",
        linewidth=2,
    )
    ax.hlines(
        -threshold_fdr,
        xmin,
        xmax,
        linestyle="--",
        colors="b",
        linewidth=2,
    )
    ax.legend()
    ax.set_title(f"{group} N2pc T-test")
    ax.set_xlabel("Time (ms)")
    ax.set_xlabel("T-stat")
    #plt.show()
    fig.savefig(os.path.join(path, f'{group}-ttest.png'))


def main():
    group_dict = {'old':['18', '19', '20', '21', '22'],
                  'thalamus': ['52', '54', '55', '56', '58'],
                  'pulvinar':['51', '53', '59', '60'],
                    'young': ['71', '72', '75', '76', '77']
                    }
    
    for group, subject_list in group_dict.items():
        try:
            X, times = get_n2pc_array_group(subject_list)
            T, reject_fdr, pval_fdr, threshold_fdr = stats_n2pc(X)
            plot_n2pc(T, times, reject_fdr, pval_fdr, threshold_fdr, group)
        except:
            print(f'Error in group {group}')
            continue
  

if __name__ == '__main__':
    main()