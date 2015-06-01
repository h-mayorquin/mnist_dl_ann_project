import numpy as np
import matplotlib.pyplot as plt
import os

#################
# Load the tuple
#################

folder = './'
name = 'rbm_tuple'
extension = '.npy'


tuple = np.load(folder + name + extension)
weights = tuple[0]  # This is the weights
samples = tuple[1]  # This is the actual sampling
vis_samples = tuple[2]  # This is the mean activation that should be ploted

# Transform them to arrays
weights = np.array(weights)
samples = np.array(samples)
vis_samples = np.array(vis_samples)

###################
# Extract relevant information
###################

# Chose the how much filters you are going to plot
Nplot = 8
hidden_indexes = np.random.choice(500, Nplot, replace=False)

# Get the last weights
data = weights[:, hidden_indexes, ...]

# Reshape the data
# data = data.reshape((Nplot, 28, 28))  # 28 is the size of the side MNIST

###################
# Plot the files
##################
if True:
    # [label.reshape((28, 28)) for label in labels_list]
    to_plot = data
    # to_plot = to_plot[0]

    # Plot parameters
    fontsize = 20
    figsize = (16, 12)
    axes_position = [0.1, 0.1, 0.8, 0.8]
    remove_axis = True
    title = 'Stress vs size of embedding space'
    xlabel = 'Dimension'
    ylabel = 'Stress'

    # Plot format
    inter = 'nearest'
    cmap = 'hot'
    cmap = 'jet'
    cmap = 'binary'

    # Save directory
    folder = './results/'
    extensions = '.pdf'
    name = 'evolving_filters'
    filename = folder + name + extensions

    # Plot here
    nplots1 = data.shape[0]
    nplots2 = data.shape[1]
    gs = plt.GridSpec(nplots1, nplots2)
    fig = plt.figure(figsize=figsize)
    axes = []

    for index1 in xrange(nplots1):
        for index2 in xrange(nplots2):

            ax = fig.add_subplot(gs[index1, index2])
            map = to_plot[index1, index2, ...]
            map = map.reshape((28, 28))
            im = ax.imshow(map, interpolation=inter, cmap=cmap)
            # ax.set_title(str(partition_set[index]) + ' clusters')
            # ax.set_aspect(1)
            axes.append(ax)

    # Remove the axis
    if remove_axis:
        for ax in axes:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    # fig.tight_layout(pad=0, w_pad=0, h_pad=0)

    plt.subplots_adjust(left=0.25, right=0.75, wspace=0.0, hspace=0)

    # Save the figure
    plt.savefig(filename)
    os.system('pdfcrop %s %s' % (filename, filename))
    plt.show()
