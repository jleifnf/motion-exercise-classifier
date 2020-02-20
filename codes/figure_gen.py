import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def ax_params(xlabel, ylabel, plt_title=None, ax=None, legend_title=None, savefig=False):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.title(plt_title)
    if ax is None:
        ax = plt.gca()
    if legend_title:
        ax.legend(title=legend_title, loc='best')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')
    if savefig:
        plt.gcf().savefig(f'{plt_title}.png', dpi=200, transparent=True, bbox_inches='tight')


def line_plot(y=None, x=None, data=None, ax=None, xlabel=None, ylabel=None,
              plt_title=None, legend_title=None, savefig=False, figsize=(8, 6)):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    y = ['x', 'y', 'z'] if y is None else [y]
    x = 'time' if x is None else x
    for c in y:
        sns.lineplot(x=x, y=c, data=data, label=c, ax=ax)
    ax.axis('tight')
    ax_params(xlabel, ylabel, plt_title=plt_title, ax=ax, legend_title=legend_title,
              savefig=savefig)


def spec_plot(S, ax=None):
    ax.imshow(10 * np.log10(S), aspect='auto')
    ax.invert_yaxis()
