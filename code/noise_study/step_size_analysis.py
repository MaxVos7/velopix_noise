import matplotlib.pyplot as plt
import numpy as np
import meta_helper


def get_step_sizes(module: int, vp: str) -> list:
    all_trim_averages = meta_helper.fetch_trim_averages(meta_helper.ALL_TRIM_LEVELS, module=module, vp=vp)
    step_sizes = []
    for j in np.arange(1, len(all_trim_averages)):
        step_size = all_trim_averages[j] - all_trim_averages[j - 1]
        step_sizes.append(step_size)
    return step_sizes


def plot_module_step_sizes(vps: list, axs: list, all_fits_in_one: bool = False):
    fits = []
    # Initial settings
    for i in np.arange(len(vps)):
        vp = vps[i]
        if vp == '0-0':
            module = 3
        else:
            module = 1
        axs[i].set_title(f'module {module} VP{vp}')
        # Fetching data
        step_sizes = get_step_sizes(module, vp)
        fit = meta_helper.fit_and_make_plot(meta_helper.ALL_TRIM_LEVELS[1:], step_sizes, np.arange(15), 2, axs[i],
                                            fit_color=meta_helper.COLORS[i])
        fits.append(fit)
    if all_fits_in_one:
        meta_helper.make_plot(axs[-1],
                              [meta_helper.ALL_TRIM_LEVELS] * len(vps),
                              list(map(lambda fit: fit(meta_helper.ALL_TRIM_LEVELS), fits)),
                              color_array=meta_helper.COLORS
                              )
        axs[-1].set_title('all asic fits')

    for ax in axs:
        ax.set_xticks(meta_helper.ALL_TRIM_LEVELS)
        ax.set_xticklabels(list(map(lambda trim: format(trim, 'x').upper(), meta_helper.ALL_TRIM_LEVELS)))
        ax.set_yticks(np.arange(4, 21, 2))
        ax.set_ylim(5, 21)


fig, axs = plt.subplots(3, 1, constrained_layout=True)
fig.suptitle('module 3 VP0-0')
# Get step characteristic for vp0-0
trim_averages = meta_helper.fetch_trim_averages(meta_helper.ALL_TRIM_LEVELS, 3, '0-0')
linear_fit = meta_helper.make_poly_fit(np.take(meta_helper.ALL_TRIM_LEVELS, [0, -1]), np.take(trim_averages, [0, -1]),
                                       1)
meta_helper.make_plot(axs[0], [], [trim_averages, linear_fit(meta_helper.ALL_TRIM_LEVELS)],
                      color_array=meta_helper.COLORS,
                      marker_array=['o']
                      )
axs[0].set_title('trim averages and linear fit')
residuals = list(
    map(lambda i: trim_averages[i] - linear_fit(meta_helper.ALL_TRIM_LEVELS[i]),
        np.arange(len(meta_helper.ALL_TRIM_LEVELS))))
meta_helper.make_plot(axs[1], [], [residuals], marker_array=['o'])
axs[1].set_title('residual of linear fit')
step_sizes_1 = get_step_sizes(3, '0-0')
meta_helper.make_plot(axs[2], [meta_helper.ALL_TRIM_LEVELS[:-1]], [step_sizes_1], marker_array=['o'])
axs[2].set_title('step sizes')
for ax in axs:
    ax.set_xticks(meta_helper.ALL_TRIM_LEVELS)
    ax.set_xticklabels(list(map(lambda trim: format(trim, 'x').upper(), meta_helper.ALL_TRIM_LEVELS)))

plt.show()
# Get average step size for vp0-1
vps = ['0-1', '0-2', '1-0', '1-1']
for i in np.arange(len(vps)):
    vp = vps[i]
    step_sizes_2 = get_step_sizes(1, vp)
    first_step_size = step_sizes_2[0]
    fit2 = np.poly1d([fit.coeffs[0], fit.coeffs[1], first_step_size - fit.coeffs[0] - fit.coeffs[1]])

    meta_helper.make_plot(axs[i],
                          [meta_helper.ALL_TRIM_LEVELS[1:], meta_helper.ALL_TRIM_LEVELS[1:]],
                          [step_sizes_2, fit2(meta_helper.ALL_TRIM_LEVELS[1:])],
                          marker_array=['o'],
                          color_array=['blue', 'orange']
                          )
# make prediction for step size vp0-1
# make plot
axs[0].set_title('trim averages + linear fit')
axs[1].set_title('residual from linear fit')
axs[2].set_title('step sizes')

plt.show()
