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


def getNoiseMeanPrediction(decTrimArray: list, trimMatrixArray: list, row: int, col: int, trimLevel: str) -> int:
    decTrimLevel = int(trimLevel, 16)

    # [0: 1200, 15: 1800, 4: 1500]
    trimDataArray = list(map(lambda trimMatrix: trimMatrix[row][col], trimMatrixArray))
    poly = np.poly1d(np.polyfit(decTrimArray, trimDataArray, 1))

    return poly(decTrimLevel)


def makeDistanceToPredictedArray(trimArray: list, trimDataArray: list, trimToPredictArray: np.array,
                                 toPredictTrim: str) -> list:
    """
    This method makes an array with the distance to the predicted value per pixel.

    :param trimArray: Give an array of trim levels, to be used for prediction (in same order as trimDataArray)
    :param trimDataArray: Give for each trim level, the matrix with means per pixel.
    :param trimToPredictArray: Give the trim matrix to predict.
    :param toPredictTrim: Give the trim level that is the to be predicted (trim level of the trimToPredictArray)
    :return: An 1D array of distance to predicted value.
    """
    distanceToPredictedArray = []
    decTrimArray = list(map(lambda trim: int(trim, 16), trimArray))
    for i in range(trimToPredictArray.shape[0]):
        for j in range(trimToPredictArray.shape[1]):
            if trimToPredictArray[i][j] > 0 and trim0Array[i][j] > 0 and trimFArray[i][j] > 0:
                continue
            else:
                noiseMeanPrediction = getNoiseMeanPrediction(decTrimArray, trimDataArray, i, j, toPredictTrim)
                distanceToPredictedArray.append(
                    trimToPredictArray[i][j] - noiseMeanPrediction)

    return distanceToPredictedArray


def showHistogram(predictionOffsetArray: list):
    hist_data = np.histogram(predictionOffsetArray,
                             np.arange(min(predictionOffsetArray), max(predictionOffsetArray), 1))
    logs = hist_data[0].astype(float)
    plt.semilogy(hist_data[1][:-1], logs, marker='.', linestyle='')
    plt.suptitle("Use trim 0, D, F to predict trim D")
    plt.xlabel("Difference of prediction and real value (current)")
    plt.ylabel("The amount of times difference is found")
    plt.show()


trim0Array = fetch_noise_mean_array('0')
trimFArray = fetch_noise_mean_array('F')
trim1Array = fetch_noise_mean_array('1')

trimDArray = fetch_noise_mean_array('D')

predictionOffset = makeDistanceToPredictedArray(['0', 'F'], [trim0Array, trimFArray], trimDArray, 'D')

showHistogram(predictionOffset)
