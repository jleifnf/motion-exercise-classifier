import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def ax_params(xlabel, ylabel, ax=None, savefig=False, **kwargs):
    """

    Args:
        xlabel:  *required
        ylabel:  *required
        ax:  current ax to modify with parameters
        savefig: False to only display, default; True to save figure wit hte
        **kwargs:
                        labels - legend labels (ex: ['x','y','z'])
                        legend_title - title for the legend
                        plt_title - title for the plot and

    Returns:

    """
    plt_title = kwargs.get('plt_title') if kwargs.get('plt_title') else 'Plot'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    if ax is None:
        ax = plt.gca()
    ax.legend(labels=kwargs.get('labels'), title=kwargs.get('legend_title'), loc='best')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')
    if savefig:
        plt.gcf().savefig(f'{plt_title}.png', dpi=200, transparent=True, bbox_inches='tight')
    else:
        plt.title(plt_title)  # only show the title when in jupyter notebook


def line_plot(data=None, ax=None, xlabel=None, ylabel=None,
              plt_title=None, legend_title=None, savefig=False, figsize=(8, 6), **kwargs):
    if ax is None:
        r, c = kwargs.get('subplots')[0:] if kwargs.get('subplots') else (1, 1)
        fig, ax = plt.subplots(r, c, figsize=figsize)
    labels, colors = (kwargs.get('labels'), ['r', 'b', 'g']) if kwargs.get('labels') else (
    ['x', 'y', 'z'], ['r', 'b', 'g'])
    for i, l in enumerate(labels):
        sns.lineplot(data=data[:, i], ax=ax, dashes=False, color=colors[i])
    #     ax.axis('tight')
    ax.set_xlim([0, data.shape[0]])  # assumed shape is [time,samples]
    ax.set_ylim([data.min() - 0.5 * data.std(), data.max() + 0.5 * data.std()])
    ax_params(xlabel, ylabel, ax=ax, savefig=savefig, **dict(legend_title=legend_title,
                                                             plt_title=plt_title, labels=labels))


def spec_plot(S, ax=None):
    """
    Plots a time-frequency spectrogram and returns in power (dB) image.
    Args:
        S: Spectrogram matrix (2D-array)
        ax:

    Returns:

    """
    ax.imshow(10 * np.log10(S), aspect='auto')
    ax.invert_yaxis()
