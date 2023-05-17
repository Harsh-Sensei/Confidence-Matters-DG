import os
import sys
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

NUM_CLASSES = None # picked from training script
NUM_EXPERTS = 6
NUM_SPARSE = 1
CLIP_MODEL = "ViT-B/32"

ROUTING = None
DOMAIN_CLASS_DICT = None
TEST_ENV = None

MOE_LAYERS = ['F']*8 + ['F', 'F', 'F', 'S']

expert_usage_dict_sparse_train = [{} for i in range(NUM_SPARSE)]
expert_usage_dict_sparse_eval = [{} for i in range(NUM_SPARSE)]

current_y = None
current_domain_y = None
curr_sparse_layer = 0

# Variables for inference of hardcoded models
ground_truth_stack = [] # stack of all the ground truth values
logit_stack = [] # stack of all logits
EXPERT_SPARSE = [] #(for determining the expert to be used) would change during inference
predicted_class_stack = []
# FIRST_SPARSE_H = True

def init_config():
    for i in range(NUM_CLASSES):
        for j in range(NUM_SPARSE):
            expert_usage_dict_sparse_train[j][i] = [0]*NUM_EXPERTS
            expert_usage_dict_sparse_eval[j][i] = [0]*NUM_EXPERTS
    for i in range(NUM_SPARSE):
        EXPERT_SPARSE.append(0)

def save_config(filename="expert_routing_classwise"):
    for i in range(NUM_SPARSE):
        filehandler = open(f"{filename}_train_{i}.pkl", 'wb')
        pickle.dump(expert_usage_dict_sparse_train[i], filehandler)
        filehandler.close()

        filehandler = open(f"{filename}_eval_{i}.pkl", 'wb')
        pickle.dump(expert_usage_dict_sparse_eval[i], filehandler)
        filehandler.close()

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# TODO : expert_usage_dict dtype changed
def draw_heatmap(filename="plots/expert_heatmap"):
    data = np.zeros(shape=(NUM_CLASSES, NUM_EXPERTS))
    for key, value in expert_usage_dict_sparse.items():
        for idx, elem in enumerate(value):
            data[key, idx] = elem
    plt.imshow(data, cmap='autumn', interpolation='nearest', aspect='auto')
    plt.xlabel("Experts")
    plt.ylabel("Classes")
    plt.title("2-D Heat Map ")
    plt.savefig(f"{filename}_1.png")

    for key, value in expert_usage_dict_sparse2.items():
        for idx, elem in enumerate(value):
            data[key, idx] = elem
    plt.imshow(data, cmap='autumn', interpolation='nearest', aspect='auto')
    plt.xlabel("Experts")
    plt.ylabel("Classes")
    plt.savefig(f"{filename}_2.png")

