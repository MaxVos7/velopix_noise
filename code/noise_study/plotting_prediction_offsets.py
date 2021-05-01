from matplotlib import pyplot as plt
import numpy as np
import offset_and_mask_handler
from matplotlib import cm


def make_prediction_offset_plot(arrays: list, trims_to_plot: list):
    fig, axs = plt.subplots(len(arrays), 1)

    if len(trims_to_plot) == 1:
        axs = [axs]

    minimum = min(map(np.min, arrays))
    maximum = max(map(np.max, arrays))

    colors = cm.get_cmap('copper', len(arrays) + 1)
    for index in range(len(trims_to_plot)):
        make_histogram(axs[index], arrays[index], trims_to_plot[index],
                       minimum, maximum, colors(index))

    axs[-1].set(xlabel='predicted value - real value (DAC current)')
    plt.show()


def make_histogram(ax: plt.axes, prediction_offset_array: list, trim_to_predict: str,
                   min: int, max: int, color: str):
    hist_data = np.histogram(prediction_offset_array,
                             np.arange(min, max, 1))
    logs = hist_data[0].astype(float)
    mean = round(np.mean(prediction_offset_array), 2)
    ax.semilogy(hist_data[1][:-1], logs, linestyle='-', color=color)
    ax.set_yticks([1, 100, 10000])
    ax.text(.9, .6, f"Trim {trim_to_predict}\nmean: {mean}",
            horizontalalignment='center',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=1),
            fontsize=8,
            color=color
            )
    ax.label_outer()
    ax.set(ylabel='count')


def make_plots(trim_levels: list, prediction_bases: list, mask_types_array: list):
    arrays_to_plot = []
    arrays_to_plot_labels = []

    for prediction_base in prediction_bases:
        for trim in trim_levels:
            for mask_types in mask_types_array:
                offset_matrix = offset_and_mask_handler.fetch_prediction_offset_matrix(trim,
                                                                                       prediction_base=prediction_base
                                                                                       )
                offset_without_zero_mean = offset_and_mask_handler.filter_masked_pixels(offset_matrix,
                                                                                        trim=trim,
                                                                                        mask_types=mask_types,
                                                                                        prediction_base=prediction_base
                                                                                        )
                arrays_to_plot.append(offset_without_zero_mean)
                arrays_to_plot_labels.append(prediction_base + '->' + trim)

    make_prediction_offset_plot(arrays_to_plot,
                                arrays_to_plot_labels)


# trim_levels = list(map(lambda value: format(value, 'x').upper(), np.arange(1,15)))
trim_levels = ['1', '5', 'A', 'E']

make_plots(trim_levels, ['0F', '04DF'], [['Dead']])
