# In this file we would like to study the behaviour per pixel.
# We want to make a prediction using trim 0 and F for a third trim level and see per pixel how well
# that prediction fits.

# First we fetch the noise mean matrix for trim 0 1 and a third trim level.
# Then we go want to be able to make a prediction per pixel.
import glob
import numpy as np
from matplotlib import pyplot as plt


def getTrimMatrix(trim_level: str) -> np.array:
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

    trimDataArray = list(map(lambda trimMatrix: trimMatrix[row][col], trimMatrixArray))

    poly = np.poly1d(np.polyfit(decTrimArray, trimDataArray, 1))

    return poly(decTrimLevel)


def checkForZeroValues(trimDataArray: list, row: int, col: int) -> bool:
    for trimData in trimDataArray:
        if trimData[row][col] == 0:
            return True

    return False


def makeDistanceToPredictedArray(trimsToPredictWith: list, trimToPredict: str) -> tuple[list, int]:
    trimsToPredictWithMatrices = list(map(lambda trim: getTrimMatrix(trim), trimsToPredictWith))
    trimToPredictMatrix = getTrimMatrix(trimToPredict)

    """
    This method makes an array with the distance to the predicted value per pixel.

    :param trimArray: Give an array of trim levels, to be used for prediction (in same order as trimDataArray)
    :param trimDataArray: Give for each trim level, the matrix with means per pixel.
    :param trimToPredictArray: Give the trim matrix to predict.
    :param toPredictTrim: Give the trim level that is the to be predicted (trim level of the trimToPredictArray)
    :return: An 1D array of distance to predicted value.
    """
    distanceToPredictedArray = []
    amountNotUsed = 0
    decTrimArray = list(map(lambda trim: int(trim, 16), trimsToPredictWith))
    for i in range(trimToPredictMatrix.shape[0]):
        for j in range(trimToPredictMatrix.shape[1]):
            if trimToPredictMatrix[i][j] != 0 and not checkForZeroValues(trimsToPredictWithMatrices, i, j):
                noiseMeanPrediction = getNoiseMeanPrediction(decTrimArray, trimsToPredictWithMatrices, i, j,
                                                             trimToPredict)
                distanceToPredictedArray.append(noiseMeanPrediction - trimToPredictMatrix[i][j])
            else:
                amountNotUsed += 1

    return (distanceToPredictedArray, amountNotUsed)


def makeHistogram(predictionOffsetArray: list, trimsToPredictWith: list, trimToPredict: str, amountNotUsed: int):
    hist_data = np.histogram(predictionOffsetArray,
                             np.arange(min(predictionOffsetArray), max(predictionOffsetArray), 1))
    logs = hist_data[0].astype(float)
    plt.semilogy(hist_data[1][:-1], logs, linestyle='-', label=f"{','.join(trimsToPredictWith)} -> {trimToPredict}")
    print(f"{','.join(trimsToPredictWith)} -> {trimToPredict}")
    print(f"mean: {round(np.mean(predictionOffsetArray), 2)}")
    print(f"points used: {len(predictionOffsetArray)}")
    print(f"points not used: {amountNotUsed}")


plt.xlabel("Predicted value - real value (current)")
plt.ylabel("The amount of times the difference is found")

(predictionOffset, amountNotUsed) = makeDistanceToPredictedArray(['0', '3', 'D', 'F'], 'D')

makeHistogram(predictionOffset, ['0', '3', 'D', 'F'], 'D', amountNotUsed)

(predictionOffset, amountNotUsed) = makeDistanceToPredictedArray(['0', '3', 'F'], 'D')

makeHistogram(predictionOffset, ['0', '3', 'F'], 'D', amountNotUsed)

(predictionOffset, amountNotUsed) = makeDistanceToPredictedArray(['0', 'F'], 'D')

makeHistogram(predictionOffset, ['0', 'F'], 'F', amountNotUsed)

plt.legend(loc='upper right')

plt.show()
