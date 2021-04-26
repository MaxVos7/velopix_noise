import offset_and_mask_handler as mask_handler

import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm


def make_mask_map(predictions_to_map: list, mask_type: str):
    outside_std_mask_matrices = list(map(
        lambda prediction_trim: mask_handler.get_mask_matrices(prediction_trim, [mask_type])[0], predictions_to_map)
    )
    outside_2std_total = np.zeros(outside_std_mask_matrices[0].shape)
    for matrix in outside_std_mask_matrices:
        outside_2std_total += matrix

    # Bitmap for outside std pixels
    bitmap = np.zeros(
        (256, 256, 4))  # create a 256x256x3 array where each pixel has an RGB value of 1-bit representation
    cmap = cm.get_cmap('summer_r', len(predictions_to_map) + 1)
    for i in np.arange(len(predictions_to_map) + 1):
        bitmap[outside_2std_total == i] = cmap(i)

    # Adding to bitmap the dead pixels.
    dead_matrices = list(map(
        lambda prediction_trim: mask_handler.get_mask_matrices(prediction_trim, ['Dead'])[0], predictions_to_map)
    )
    dead_total = np.zeros(dead_matrices[0].shape)
    for dead_matrix in dead_matrices:
        dead_total += dead_matrix

    # Creating legend
    patches = [mpatches.Patch(color=cmap(0),
                              label=f"Inside 2 std: {np.count_nonzero(outside_2std_total == 0) - np.count_nonzero(dead_total)}")]
    for i in np.arange(1, len(predictions_to_map) + 1):
        patches.append(mpatches.Patch(
            color=cmap(i),
            label=f"Outside 2 std in {i} prediction{'s' if i > 1 else ''}: {np.count_nonzero(outside_2std_total == i)}")
        )
    patches.append(mpatches.Patch(color='white',
                                  label=f"Tot Outside 2 standard deviations: {np.count_nonzero(outside_2std_total)}"))

    bitmap[dead_total > 0] = matplotlib.colors.to_rgba('black')
    patches.append(mpatches.Patch(color='black', label=f"Dead: {np.count_nonzero(dead_total)}"))

    plt.legend(bbox_to_anchor=(1.042, 0.3), loc=2, borderaxespad=0, handles=patches,
               frameon=False, handlelength=1.25)
    plt.axis([-2, 257, -2, 257])  # to better show the borders
    plt.xlabel("Pixel x coordinate")
    plt.ylabel("Pixel y coordinate")
    plt.imshow(bitmap)
    plt.show()


make_mask_map(['1', '3', 'D'], 'Outside_2Std_Not_Dead')
