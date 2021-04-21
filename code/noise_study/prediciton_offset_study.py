import glob

from matplotlib import pyplot as plt
import numpy as np

MASK_VALUE = 1234


def make_histogram(ax: plt.axes, prediction_offset_array: list, trim_to_predict: str,
                   min: int, max: int, color: str):
    hist_data = np.histogram(prediction_offset_array,
                             np.arange(min, max, 1))
    logs = hist_data[0].astype(float)
    mean = round(np.mean(prediction_offset_array), 2)
    ax.semilogy(hist_data[1][:-1], logs, linestyle='-', color=color)
    ax.set_yticks([1, 100, 10000])
    ax.text(max - 40, 500, f"Trim {trim_to_predict}\nmean: {mean}", fontsize=10,
            color=color)
    ax.label_outer()
    ax.set(ylabel='count')


def fetch_prediction_offset_matrix(trim) -> (np.ndarray, np.ndarray):
    files = glob.glob(f"../../data/prediction_offset_matrices/Prediction_Offset_0F_to_{trim}.csv")
    if len(files) < 1:
        raise FileNotFoundError("There is no prediction offset matrix for the given trim.")

    prediction_matrix = np.loadtxt(files[0], delimiter=',')

    files = glob.glob(f"../../data/prediction_offset_matrices/Masked_0F_to_{trim}.csv")
    if len(files) < 1:
        raise FileNotFoundError("There is no mask matrix for the given trim.")

    mask_matrix = np.loadtxt(files[0], delimiter=',')

    return prediction_matrix, mask_matrix


def make_prediction_offset_plot(offset_arrays: list, trims_to_plot: list):
    fig, axs = plt.subplots(len(offset_arrays), 1)

    if len(trims_to_plot) == 1:
        axs = [axs]

    minimum = np.amin(offset_arrays)
    maximum = np.amax(offset_arrays)

    colors = ['red', 'blue', 'green', 'orange', 'magenta', 'cyan']
    for index in range(len(trims_to_plot)):
        make_histogram(axs[index], offset_arrays[index], trims_to_plot[index],
                       minimum, maximum, colors[index])

    axs[-1].set(xlabel='predicted value - real value (DAC current)')
    plt.show()


def find_pixels_outside_std(matrix: np.ndarray, stdAmount: int = 1) -> np.ndarray:
    (width, height) = matrix.shape
    faulty_pixels = np.zeros((width, height))
    mean = matrix.mean()
    std = matrix.std()
    for i in np.arange(width):
        for j in np.arange(height):
            if matrix[i][j] > mean + std * stdAmount or matrix[i][j] < mean - std * stdAmount:
                faulty_pixels[i][j] = 1

    return faulty_pixels


def make_filtered_offset_array(offset_matrix: np.ndarray, faulty_pixels: np.ndarray = None):
    result = []
    for i in np.arange(offset_matrix.shape[0]):
        for j in np.arange(offset_matrix.shape[1]):
            if offset_matrix[i][j] != MASK_VALUE and faulty_pixels[i][j] == 0:
                result.append(offset_matrix[i][j])

    return result


offset_matrix, mask_matrix = fetch_prediction_offset_matrix('1')

check = offset_matrix.copy()[mask_matrix == 0]
print(offset_matrix.size, len(check))
exit()

faulty_pixels = find_pixels_outside_std(offset_matrix, 1)
filtered_offset_array = make_filtered_offset_array(offset_matrix, faulty_pixels)

print(filtered_offset_array)

make_prediction_offset_plot([filtered_offset_array], ['1'])
