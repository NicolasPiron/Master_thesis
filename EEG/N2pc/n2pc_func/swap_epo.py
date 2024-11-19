import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main_swap_epo(subject_id, indir):
    
    fname = os.path.join(indir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif')
    epochs = mne.read_epochs(fname)
    epochs.info['bads'] = [] # remove bad channels so that the data can be swapped fully
    swapped_epochs = swap_epo(epochs)
    fig1 = plot_swp_po(epochs, swapped_epochs)
    fig2 = plot_evk_swp(epochs, swapped_epochs)
    fig3 = plot_topo_swp(epochs, swapped_epochs)
    fig4 = plot_events(epochs, swapped_epochs)
    figs = plot_cond(epochs, swapped_epochs)
    
    outdir = os.path.join(indir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', 'swapped_epochs')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print(f'--- Saving swapped epochs for subject {subject_id} ---')
    swapped_epochs.save(os.path.join(outdir, f'sub-{subject_id}-cleaned_epochs-N2pc-swp.fif'), overwrite=True)
    fig1.savefig(os.path.join(outdir, f'sub-{subject_id}-swp_po.png'))
    fig2.savefig(os.path.join(outdir, f'sub-{subject_id}-evk_swp.png'))
    fig3.savefig(os.path.join(outdir, f'sub-{subject_id}-topo_swp.png'))
    fig4.savefig(os.path.join(outdir, f'sub-{subject_id}-events_swp.png'))
    for i, fig in enumerate(figs):
        fig.savefig(os.path.join(outdir, f'sub-{subject_id}-cond_{i}.png'))
    print('--- Done ---')
    
    return None

def swap_epo(epochs):
    ''' returns the swapped epoched data'''

    swp_evt = swap_evt(epochs)
    swp_data = swap_data(epochs)
    return mne.EpochsArray(swp_data, epochs.info, events=swp_evt,
                        event_id=epochs.event_id, tmin=epochs.tmin, baseline=epochs.baseline)
    
def swap_data(epochs):
    ''' returns the swapped epoched data'''
    
    lch, rch, _ = get_ch_side(epochs)
    data = epochs.get_data(copy=True)
    swp_data = data.copy()
    swp_data[:, rch, :] = data[:, lch, :]
    swp_data[:, lch, :] = data[:, rch, :]

    return swp_data

def get_ch_side(epochs):
    '''returns the indices of the channels in the left and right hemisphere'''

    ch_names = epochs.ch_names
    ch_names = ch_names[:64] # only keep the EEG channels
    if 'Status' in ch_names: # shouldn't be necessary if indices are cropped correctly
        ch_names.remove('Status')
    LCh = []
    RCh = []
    MCh = []
    for i, ch in enumerate(ch_names):
        if str(ch[-1]) == 'z':
            MCh.append(i)
        elif int(ch[-1]) % 2 == 0:
            RCh.append(i)
        elif int(ch[-1]) %2 != 2:
            LCh.append(i)

    return LCh, RCh, MCh

def swap_evt(epochs):
    ''' returns the swapped events'''

    elist = epochs.events 
    swp_evt = elist.copy()
    map_dict = {1:2, 2:1, 3:4, 4:3, 5:6, 6:5, 7:8, 8:7}
    swp_evt[:,2] = [map_dict[x] for x in swp_evt[:,2]]
    
    return swp_evt

############################################################################################
# Sanity check functions - visually verify that the swapping works as expected
############################################################################################

def plot_swp_po(epochs, swp_epo):
    ''' plots the first trial of the original and swapped data for PO7 and PO8'''

    po7 = epochs.copy().get_data(picks='PO7')
    po8 = epochs.copy().get_data(picks='PO8')
    swp_po7 = swp_epo.copy().get_data(picks='PO7')
    swp_po8 = swp_epo.copy().get_data(picks='PO8')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].plot(po7[0,0,:], label='PO7')
    axes[0].plot(swp_po8[0,0,:], label='PO8 in swapped data')
    axes[0].legend()
    axes[0].set_title('original PO7 and swapped PO8')

    axes[1].plot(po8[0,0,:], label='PO8')
    axes[1].plot(swp_po7[0,0,:], label='PO7 in swapped data')
    axes[1].legend()
    axes[1].set_title('original PO8 and swapped PO7')
    plt.suptitle('Comparison of original and swapped data - 1st trial')
    plt.tight_layout()
    
    return fig

def plot_evk_swp(epochs, swp_epo):
    ''' plots the average of the original and swapped data for the left and right channels'''

    lch, rch, _ = get_ch_side(epochs)
    t = epochs.times

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    epochs.average().plot(picks=lch, axes=axes[0, 0], show=False)
    axes[0, 0].set_title('original data, left chans')
    swp_epo.average().plot(picks=rch, axes=axes[0, 1], show=False)
    axes[0, 1].set_title('swapped data, right chans')

    epochs.average().plot(picks=rch, axes=axes[1, 0], show=False)
    axes[1, 0].set_title('original data, right chans')
    swp_epo.average().plot(picks=lch, axes=axes[1, 1], show=False)
    axes[1, 1].set_title('swapped data, left chans')
    plt.suptitle('Comparison of original and swapped data')
    plt.tight_layout()

    return fig

def plot_topo_swp(epochs, swp_epo):
    ''' plots the topomap of the average of the original and swapped data'''

    evk = epochs.average()
    swp_evk = swp_epo.average()

    fig, ax = plt.subplots(2, 2, figsize=(5, 5), gridspec_kw={'width_ratios': [5, 0.1]})
    evk.plot_topomap(times=[0.2], axes=(ax[0, 0], ax[0, 1]), show=False)
    swp_evk.plot_topomap(times=[0.2], axes=(ax[1, 0], ax[1, 1]), show=False)
    plt.suptitle('Comparison of original and swapped data')
    plt.tight_layout()

    return fig

def plot_events(epochs, swp_epo):
    ''' plots the events of the original and swapped data'''

    elist = epochs.events
    swp_elist = swp_epo.events
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(elist[:, 0], elist[:, 2], 'o', alpha=0.5, label='original')   
    ax.plot(swp_elist[:, 0], swp_elist[:, 2], 'x', label='swapped')
    ax.hlines([2.5, 4.5, 6.5], elist[0, 0], elist[-1, 0], color='grey', alpha=0.5, linestyle='--')
    ax.set_title('Comparison of original and swapped events')
    ax.set_xlabel('time')
    ax.set_ylabel('event type')
    ax.legend()
    plt.tight_layout()
    
    return fig

def plot_cond(epochs, swap_epo):
    ''' Plot the original and swapped evoked response for each condition'''

    dict_cond = {'dis_top/target_l':'dis_top/target_r',
                 'dis_top/target_r':'dis_top/target_l',
                 'dis_bot/target_l':'dis_bot/target_r',
                 'dis_bot/target_r':'dis_bot/target_l',
                 'no_dis/target_l':'no_dis/target_r',
                 'no_dis/target_r':'no_dis/target_l',
                 'dis_right/target_l':'dis_left/target_r',
                 'dis_left/target_r':'dis_right/target_l'}

    figs = []
    for cond, swp_cond in dict_cond.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        epochs[cond].average().plot(axes=ax1, show=False)
        swap_epo[swp_cond].average().plot(axes=ax2, show=False)
        plt.suptitle(f'{cond} and swapped {swp_cond}')
        plt.tight_layout()
        figs.append(fig)
    
    return figs