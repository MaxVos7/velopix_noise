# Subplots showing:
# the points together with linear fit between 0 and F
# the points subtracted with the linear trend
import matplotlib.pyplot as plt
import meta_helper
import numpy as np
from scipy.stats import chisquare


def find_best_4_trims_to_fit(trim_averages: list, trim_levels: list) -> tuple[list, float]:
    polynom_degree = 3
    fits = []
    min_value = None
    min_index = [None, None, None, None]
    chisquare_array = np.zeros((16, 16, 16, 16))
    for i in np.arange(0, 16):
        for j in np.arange(0, 16):
            for k in np.arange(0, 16):
                for l in np.arange(0, 16):
                    if i in (j, k, l) or j in (k, l) or k == l:
                        continue
                    trims_to_fit = [i, j, k, l]
                    trims_to_fit_averages = np.take(trim_averages, trims_to_fit)

                    fit = np.poly1d(np.polyfit(trims_to_fit, trims_to_fit_averages, polynom_degree))
                    fits.append(fit)
                    c2 = chisquare(trim_averages, list(map(lambda trim_level: fit(trim_level), trim_levels)))[0]
                    chisquare_array[i][j][k][l] = c2
                    if min_value is None or (0 < c2 < min_value):
                        min_value = c2
                        min_index = [i, j, k, l]
    return min_index, min_value


def calculate_full_fit_prediction(residual_fit: np.poly1d, trim_levels: list, trim_averages: list):
    if len(trim_levels) != 2:
        raise Exception("Please give exactly 2 trim levels.")
    if len(trim_averages) != 2:
        raise Exception("Please give exactly 2 trim averages.")
    trim1 = trim_levels[0]
    trim2 = trim_levels[1]
    i = ((trim_averages[0] - trim_averages[1]) - (residual_fit(trim1) - residual_fit(trim2))) / (trim1 - trim2)
    j = trim_averages[0] - residual_fit(trim1) - i * trim1
    predicted_fit = residual_fit
    predicted_fit.coeffs[-1] += j
    predicted_fit.coeffs[-2] += i
    return predicted_fit


def plot_all_residuals(flat_axs: np.ndarray, vps: list):
    all_residual_fits = []
    for i in np.arange(len(vps)):
        vp = vps[i]
        if vp == '0-0':
            module = 3
        else:
            module = 1

        trim_averages = meta_helper.fetch_trim_averages(meta_helper.ALL_TRIM_LEVELS, module, vp)
        residual = extract_residual(trim_averages)
        residual_fit = meta_helper.fit_and_make_plot(meta_helper.ALL_TRIM_LEVELS,
                                                     residual,
                                                     meta_helper.ALL_TRIM_LEVELS,
                                                     3, flat_axs[i],
                                                     fit_color=meta_helper.COLORS[i],
                                                     point_color=meta_helper.COLORS[i]
                                                     )
        all_residual_fits.append(residual_fit)
        flat_axs[i].set_title(f'module {module} VP{vp}')

    meta_helper.make_plot(flat_axs[-1],
                          [meta_helper.ALL_TRIM_LEVELS] * len(all_residual_fits),
                          list(map(lambda fit: fit(meta_helper.ALL_TRIM_LEVELS), all_residual_fits)),
                          color_array=meta_helper.COLORS
                          )

    for ax in flat_axs:
        ax.set_xticks(meta_helper.ALL_TRIM_LEVELS)
        ax.set_xticklabels(list(map(lambda trim: format(trim, 'x').upper(), meta_helper.ALL_TRIM_LEVELS)))
        ax.set_ylim(10, -7)

    plt.show()


def extract_residual(trim_averages):
    linear_fit = meta_helper.make_poly_fit(np.take(meta_helper.ALL_TRIM_LEVELS, [0, -1]),
                                           np.take(trim_averages, [0, -1]), 1)
    residual = list(
        map(lambda i: trim_averages[i] - linear_fit(meta_helper.ALL_TRIM_LEVELS[i]),
            np.arange(len(meta_helper.ALL_TRIM_LEVELS))))
    return residual


base_trim_averages = meta_helper.fetch_trim_averages(meta_helper.ALL_TRIM_LEVELS, 3, '0-0')
base_residual = extract_residual(base_trim_averages)
base_residual_fit = meta_helper.make_poly_fit(meta_helper.ALL_TRIM_LEVELS, base_residual, 3)

vps = ['0-1', '0-2', '1-0', '1-1']
fig, axs = plt.subplots(4, 1, constrained_layout=True)
flat_axs = axs.flatten()
measured_levels = [0, 7]

for i in np.arange(len(vps)):
    vp = vps[i]
    all_trim_averages = meta_helper.fetch_trim_averages(meta_helper.ALL_TRIM_LEVELS, 1, vp)
    measured_trim_averages = np.take(all_trim_averages, measured_levels)
    full_fit_prediction = calculate_full_fit_prediction(base_residual_fit, measured_levels, measured_trim_averages)
    full_fit_c2 = meta_helper.cal_c2(meta_helper.ALL_TRIM_LEVELS, all_trim_averages, full_fit_prediction)
    linear_fit = meta_helper.make_poly_fit(
        np.take(meta_helper.ALL_TRIM_LEVELS, [0, -1]),
        np.take(all_trim_averages, [0, -1]), 1)
    linear_fit_c2 = meta_helper.cal_c2(meta_helper.ALL_TRIM_LEVELS, all_trim_averages, linear_fit)
    meta_helper.make_plot(flat_axs[i], [],
                          [all_trim_averages, full_fit_prediction(meta_helper.ALL_TRIM_LEVELS)],
                          marker_array=['o'],
                          color_array=[meta_helper.COLORS[i], meta_helper.COLORS[i]],
                          patch_labels=[f'linear c2: {linear_fit_c2:.4}\nfit c2: {full_fit_c2:.4}']
                          )
    flat_axs[i].set_title(f'module 1 VP{vp}')
plt.show()