# In this file we would like to study the behaviour per pixel.
# We want to make a prediction using trim 0 and F for a third trim level and see per pixel how well
# that prediction fits.

# First we fetch the noise mean matrix for trim 0 1 and a third trim level.
# Then we go want to be able to make a prediction per pixel.
import glob
import numpy as np
from matplotlib import pyplot as plt


def fetch_noise_mean_array(trim_level: str) -> np.array:
    """
    This method fetches the noise mean array for a specific trim level.
    For this it uses the "Module1_VP0-0_Trim*_Noise_mean.csv file.
    """
    files = glob.glob(f"../../data/Module1_VP0-0_Trim{trim_level}_Noise_Mean.csv")
    if (len(files) < 1):
        raise FileNotFoundError(f"No file found for Noise mean of trim level {trim_level}.")
    else:
        file = files[0]

    return np.array(np.loadtxt(file, delimiter=',', dtype=int))


def getNoiseMeanPrediction(trim0Array: np.array, trimFArray: np.array, row: int, col: int, trimLevel: str) -> int:
    decTrimLevel = int(trimLevel, 16)

    trim0Value = trim0Array[row][col]
    trimFValue = trimFArray[row][col]
    return (trimFValue - trim0Value) * (decTrimLevel / 15) + trim0Value


def makeDistanceToPredictedArray(trim0Array: np.array, trimFArray: np.array, trimToPredictArray: np.array,
                                 toPredictTrim: str) -> np.array:
    width = trimToPredictArray.shape[0]
    height = trimToPredictArray.shape[1]

    distanceToPredictedArray = []
    """"
    This method makes an array with the distance to the predicted value per pixel.
    """
    for i in range(trimToPredictArray.shape[0]):
        for j in range(trimToPredictArray.shape[1]):
            distanceToPredictedArray.append(trimToPredictArray[i][j] - getNoiseMeanPrediction(trim0Array, trimFArray,
                                                                        i, j, toPredictTrim))

    return distanceToPredictedArray


trim0Array = fetch_noise_mean_array('0')
trimFArray = fetch_noise_mean_array('F')
trimDArray = fetch_noise_mean_array('D')

predictionOffset = makeDistanceToPredictedArray(trim0Array, trimFArray, trimDArray, 'D')

hist_data = np.histogram(predictionOffset, np.arange(-10, 1, 0.1))
logs = hist_data[0].astype(float)
plt.plot(hist_data[1][:-1], logs, marker='.', linestyle='', linewidth=2)
plt.show()
