import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import chisquare

COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
ALL_TRIM_LEVELS = np.arange(0, 16)


def fetch_trim_averages(trim_levels: list, module: int = 1, vp: str = '0-0') -> [int]:
    trim_averages = []
    for trim in list(map(lambda trim_level: format(trim_level, 'x').upper(), trim_levels)):
        files = glob.glob(f"../../data/Module{module}_VP{vp}_Trim{trim}_Noise_Mean.csv")
        if len(files) < 1:
            raise FileNotFoundError(f"No file found for trim {trim}.")

        trim_matrix = np.loadtxt(files[0], delimiter=',')

        trim_averages.append(trim_matrix[trim_matrix != 0].mean())

    return trim_averages


def make_plot(ax: plt.axes, x_value_array: list, y_value_array: list, marker_array=None,
              color_array=None,
              patch_labels=None):
    if marker_array is None:
        marker_array = []

    if patch_labels is None:
        patch_labels = []

    if color_array is None:
        color_array = []

    patches = []
    for i in np.arange(len(y_value_array)):
        if len(x_value_array) <= i:
            x_values = ALL_TRIM_LEVELS
        else:
            x_values = x_value_array[i]

        marker = marker_array[i] if len(marker_array) > i else '-'
        color = color_array[i] if len(color_array) > i else 'blue'
        ax.plot(x_values, y_value_array[i], marker, color='tab:' + color)
        ax.set_xticks([])
        if len(patch_labels) > i:
            patches.append(mpatches.Patch(color='tab:' + color, label=patch_labels[i]))

        ax.legend(bbox_to_anchor=(1.042, 1), loc=2, borderaxespad=0, handles=patches,
                  frameon=False, handlelength=1.25)


def cal_c2(all_trim_levels: list, all_trim_averages: list, fit: np.poly1d) -> float:
    return chisquare(all_trim_averages, list(map(lambda trim_level: fit(trim_level), all_trim_levels)))[0]


def make_poly_fit(trim_levels: list, trim_averages: list, degree: int) -> np.poly1d:
    return np.poly1d(
        np.polyfit(trim_levels, trim_averages, degree)
    )


def pretty_poly(poly: np.poly1d):
    order = poly.order
    string = ''
    for i in np.arange(0, order + 1):
        string += f'{poly.coeffs[i]:.4}'
        if order - i != 0:
            string += rf'$x^{order - i}$'
        if i != order:
            string += ' + '
    return string


def fit_and_make_plot(x_array: list, y_array: list, fit_indices: list, pol_deg: int,
                      ax: plt.axes = None, fit_color: str = None, point_color: str = None) -> np.poly1d:
    trims_to_fit = np.take(x_array, fit_indices)
    averages_to_fit = np.take(y_array, fit_indices)
    fit = make_poly_fit(trims_to_fit, averages_to_fit, pol_deg)
    c2 = cal_c2(x_array, y_array, fit)
    if ax is not None:
        if fit_color is None: fit_color = 'orange'
        if point_color is None: point_color = 'blue'
        make_plot(ax,
                  [x_array, x_array, trims_to_fit],
                  [fit(x_array), y_array, averages_to_fit],
                  color_array=[fit_color, point_color, fit_color],
                  marker_array=['-', 'o', 'o'],
                  # patch_labels=[f'chi-square: {c2:.4}\n' + rf'{pretty_poly(fit)}']
                  )
    return fit
