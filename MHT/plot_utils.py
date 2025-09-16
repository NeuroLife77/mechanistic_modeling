import numpy as np
import matplotlib.pyplot as plt
from pystorm import mnt
from matplotlib import pyplot as plt
from torch import logical_and as torch_logical_and
from types import FunctionType as _FunctionType
from copy import deepcopy as dcp

def remove_diag(conn,return_numpy=False):
    return_format = mnt.ensure_torch
    if return_numpy:
        return_format = mnt.ensure_numpy
    return return_format((mnt.ensure_torch(conn) * (mnt.ones(conn.shape[-2],conn.shape[-1])-mnt.eye(conn.shape[-2]))))
def nan_diag(conn,return_numpy=False):
    return_format = mnt.ensure_torch
    if return_numpy:
        return_format = mnt.ensure_numpy
    return return_format(mnt.ensure_torch(conn) + mnt.zeros_like(mnt.ensure_torch(conn)).fill_diagonal_(np.nan))

def set_plot_cmap(ax,bounds):
    if isinstance(bounds, list) and len(bounds)==3:
        space = mnt.linspace(*bounds)
    else:
        space = bounds
    ax.set_prop_cycle('color', plt.cm.viridis(space))

def plot_PSD_grid(
                    psds,
                    freq_range = [2,50],
                    freq_res = 0.5, freq_start = 0, plot_mean = True, cmap_order = None,
                    col_size = 3.5, row_size = 14, dpi = 150,
                    title_pattern = "PSD ", row_title_prefix = None,
                    y_label = "Power", x_label = "Freq [Hz]", plot_fn = "plot",
                    ylim = None, xlim = None,
                    suptitle = None, 
                    titles = None,
                    colormap = "viridis",
                    **plotkwargs
    ):
    if "mean_lw" in plotkwargs:
        mean_lw = plotkwargs["mean_lw"]
        plotkwargs.pop("mean_lw")
    else:
        mean_lw = 1.5
    psd_shape = [len(psds),len(psds[0]),len(psds[0][0])]
    try:
        psd_shape.append(len(psds[0][0][0]))
    except:
        pass
    freqs = mnt.linspace(freq_start*freq_res,(psd_shape[-1]-1)*freq_res+freq_start,psd_shape[-1])
    freq_mask = torch_logical_and(freqs>freq_range[0],freqs<freq_range[1])
    if type(plot_fn) != _FunctionType:
        if plot_fn == "semilogy":
            plot_fn = plt.semilogy
        elif plot_fn == "semilogx":
            plot_fn = plt.semilogx
        elif plot_fn == "loglog":
            plot_fn = plt.loglog
        else:
            plot_fn = plt.plot
    fig = plt.figure(figsize=(row_size, col_size*psd_shape[1]), dpi = dpi)
    counter = 1
    for j in range(psd_shape[1]):
        if isinstance(row_title_prefix,list):
            title_prefix = row_title_prefix[j] + str(j)
        elif row_title_prefix is None:
            title_prefix = ""
        else:
            title_prefix = "Row " + str(j)
        for i in range(psd_shape[0]):
            if isinstance(y_label,list):
                ylabel=y_label[i]
            else:
                ylabel=y_label
            if isinstance(x_label,list):
                xlabel=x_label[i]   
            else:
                xlabel=x_label
            if isinstance(title_pattern,list):
                title = title_prefix + title_pattern[i]
            else:
                title= title_prefix + title_pattern + str(i)
            if titles is not None:
                title = titles[j][i]
            ax = plt.subplot(psd_shape[1],psd_shape[0],counter)
            if cmap_order is not None:
                
                if colormap == "bwr":
                    ax.set_prop_cycle('color', plt.cm.bwr(mnt.linspace(0,1,psds[i][j].shape[0])))
                elif colormap == "brg":
                    ax.set_prop_cycle('color', plt.cm.brg(mnt.linspace(0,1,psds[i][j].shape[0])))
                else:
                    ax.set_prop_cycle('color', plt.cm.viridis(mnt.linspace(0,1,psds[i][j].shape[0])))
                
                ordering = cmap_order[i][j]
            else:
                ordering = mnt.arange(psds[i][j].shape[0])
            counter+=1
            plot_fn(freqs[freq_mask],mnt.ensure_numpy(psds[i][j][...,freq_mask].T[:,ordering]),**plotkwargs)
            if plot_mean:
                mean_kwargs = dcp(plotkwargs)
                mean_kwargs["lw"] = mean_lw
                plot_fn(freqs[freq_mask],mnt.ensure_numpy(psds[i][j][:,freq_mask]).mean(0),"--k",**mean_kwargs)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.ylim(ylim)
            plt.xlim(xlim)
            plt.title(title)
    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

def plot_AEC_grid(
                    aecs,
                    col_size = 3.5, row_size = 14, dpi = 150,
                    title_pattern = "AEC ", row_title_prefix = None,
                    y_label = "ROI", x_label = "ROI", plot_fn = "imshow",
                    ylim = None, xlim = None,xticklabels=None,yticklabels=None,
                    suptitle = None, ylabel_header = None,
                    titles = None,**plotkwargs
    ):
    aecs_shape = [len(aecs),len(aecs[0]),len(aecs[0][0])]
    try:
        aecs_shape.append(len(aecs[0][0][0]))
    except:
        pass
    
    plot_fn = plt.imshow
    fig = plt.figure(figsize=(row_size, col_size*aecs_shape[1]), dpi = dpi)
    counter = 1
    for j in range(aecs_shape[1]):
        if isinstance(row_title_prefix,list):
            title_prefix = row_title_prefix[j] + str(j)
        elif row_title_prefix is None:
            title_prefix = ""
        else:
            title_prefix = "Row " + str(j)
        for i in range(aecs_shape[0]):
            if isinstance(y_label,list):
                ylabel=y_label[i]
            else:
                ylabel=y_label
            if isinstance(x_label,list):
                xlabel=x_label[i]   
            else:
                xlabel=x_label
            if i==0 and ylabel_header is not None:
                if isinstance(ylabel_header,list):
                    ylabel = ylabel_header[j] + ylabel
                else:
                    ylabel = ylabel_header + ylabel
            if isinstance(title_pattern,list):
                title = title_prefix + title_pattern[i]
            else:
                title= title_prefix + title_pattern + str(i)
            if titles is not None:
                title = titles[j][i]
            ax = plt.subplot(aecs_shape[1],aecs_shape[0],counter)
            counter+=1
            npaec=mnt.ensure_numpy(aecs[i][j])
            plot_fn(nan_diag(npaec),**plotkwargs)
            plt.colorbar()
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if yticklabels is not None:
                n_ticks = len(yticklabels)
                tick_locks = np.linspace(0,npaec.shape[-1]-1,n_ticks,dtype=int)
                plt.yticks(tick_locks,yticklabels)
            if xticklabels is not None:
                n_ticks = len(xticklabels)
                tick_locks = np.linspace(0,npaec.shape[-2]-1,n_ticks,dtype=int)
                plt.xticks(tick_locks,xticklabels)
            plt.ylim(ylim)
            plt.xlim(xlim)
            plt.title(title)
    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

def stretch_range(min_max,min_max_squeeze):
    new_min = abs(min_max[0]*min_max_squeeze[0]) * (-1*int(min_max[0]<0)+ int(min_max[0]>=0))
    new_max = abs(min_max[1]*min_max_squeeze[1]) * (-1*int(min_max[1]<0)+ int(min_max[1]>=0))
    return [new_min,new_max]


def plot_gamma_single(freqs_gamma,gamma,ax,n_samples, xlabel = "Freq (Hz)",**kwargs):
    if len(gamma.shape) == 2 and gamma.shape[-1] == len(freqs_gamma): 
        gamma_plot = gamma.T
    else:
        gamma_plot = gamma
    plt.plot(freqs_gamma,gamma_plot, lw = 1 - 0.75*min(1,n_samples/200))
    plt.xlabel(xlabel)
    return ""
def plot_gamma_dist_single(freqs_gamma,gamma,ax, xlabel = "Freq (Hz)"):
    std_gamma = gamma.std(0)
    mean_gamma = gamma.mean(0)
    gamma_rg = [mean_gamma-std_gamma,mean_gamma+std_gamma]
    gamma_rg[0] = np.where(gamma_rg[0]<0,0,gamma_rg[0])
    plt.plot(freqs_gamma,mean_gamma,"-g")
    plt.fill_between(freqs_gamma,gamma_rg[0],gamma_rg[1],color="g",alpha=0.1)
    plt.xlabel(xlabel)
    return ""


def plot_gamma_coupl(
        freqs_gamma,
        gamma,ax,
        n_samples,
        summary = "mean", 
        label = "", 
        xlabel = "Freq (Hz)",
        color=None,
        show_band_bins=[8,12.5],
    ):
    reshaped_gamma = gamma#.reshape((gamma.shape[0],-1,freqs_gamma.shape[0]))
    if isinstance(summary,str) and summary == "mean":
        summary_gamma = reshaped_gamma.mean(1)
        summary_name = summary
    elif isinstance(summary,list) and summary[0] == "index":
        summary_gamma = reshaped_gamma[:,summary[1],:]
        summary_name = f"{summary[0]} {summary[1]}"
    plotkwargs = {}
    if color is not None:
        plotkwargs["color"] = color
    plt.plot(freqs_gamma,summary_gamma.T, lw = 1 - 0.75*min(1,n_samples/200),label=label,**plotkwargs)
    if show_band_bins is not None:
        plt.plot([show_band_bins[0],show_band_bins[0]],[0,summary_gamma.max()],"--k",lw=0.5)
        plt.plot([show_band_bins[1],show_band_bins[1]],[0,summary_gamma.max()],"--k",lw=0.5)

    plt.xlabel(xlabel)
    return " " + "Node "+summary_name
    

def plot_gamma_dist_coupl(freqs_gamma,gamma,ax, xlabel = "Freq (Hz)"):
    reshaped_gamma = gamma#.reshape((gamma.shape[0],-1,freqs_gamma.shape[0]))
    std_gamma = reshaped_gamma.std(0)
    mean_gamma = reshaped_gamma.mean(0)
    gamma_rg = [mean_gamma-std_gamma,mean_gamma+std_gamma]
    gamma_rg[0] = np.where(gamma_rg[0]<0,0,gamma_rg[0])
    for node in range(mean_gamma.shape[0]):
        plt.plot(freqs_gamma,mean_gamma[node],f"-C{node}",label = f"Node {node}")
        plt.fill_between(freqs_gamma,gamma_rg[0][node],gamma_rg[1][node],color=f"C{node}",alpha=0.1)
    if mean_gamma.shape[0]<5:
        plt.legend()
    plt.xlabel(xlabel)
    return "nodes"