import matplotlib.colors
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def fetch_noise_mean_matrices(trims: list) -> list[np.ndarray]:
    matrices = []
    for trim in trims:
        matrices.append(
            np.loadtxt(glob.glob(f"../../data/Module1_VP0-0_Trim{trim}_Noise_Mean.csv", )[0], delimiter=',', dtype=int))
    return matrices


def find_zero_mean_pixels(matrix) -> list[np.ndarray]:
    func = np.vectorize(lambda value: 0 if value > 0 else 1)
    return func(matrix)


noise_mean_matrices = fetch_noise_mean_matrices(['0', '1', '3', 'D', 'F'])

zero_mean_matrices = list(map(find_zero_mean_pixels, noise_mean_matrices))

result = np.zeros(noise_mean_matrices[0].shape)
for matrix in zero_mean_matrices:
    result += matrix

bitmap = np.zeros(
    (256, 256, 3))  # create a 256x256x3 array where each pixel has an RGB value of 1-bit representation
bitmap[result == 1] = [0, 0, 0]  # Cat A: black
bitmap[result == 2] = [1, 0, 0]  # Cat B: red
bitmap[result == 3] = [0, 0, 1]  # Cat C: blue
bitmap[result == 4] = [0, 0.502, 0]  # Cat D: green
bitmap[result == 5] = [1, 0.647, 0]  # Cat E: orange -> from matplotlib.color.to_rgba("orange")
bitmap[result == 0] = [1, 1, 1]  # No mask: light grey

plt.imshow(bitmap)
plt.show()