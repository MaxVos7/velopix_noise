# This file calculates the average of the noise mean for every trim level that is available.
import glob
import numpy as np
import matplotlib.pyplot as plt


def getDecimaleTrimLevel(file: str) -> int:
    return int(file.split('_')[-3][-1], 16)


def getMeanAveragePerTrim() -> np.array:
    files = glob.glob('../data/Module1_VP0-0_Trim*_Noise_Mean.csv')

    trimLevelArray = []
    meanAvarageArray = []

    for file in files:
        meanAverage = np.loadtxt(file, delimiter=',', dtype=int).mean()
        decimalTrimLevel = getDecimaleTrimLevel(file)  # trim level in decimal value (0,1,..,14,15)
        trimLevelArray.append(decimalTrimLevel)
        meanAvarageArray.append(meanAverage)

    return [trimLevelArray, meanAvarageArray]


(trimLevelArray, meanAverageArray) = getMeanAveragePerTrim()

poly = np.poly1d(np.polyfit(trimLevelArray, meanAverageArray, 1))

print(poly, poly(0), poly(1))

plt.ylabel("Average of noise means")
plt.xlabel("Trim level")
plt.xticks(np.arange(16), list(map(lambda dec: np.base_repr(dec, 16), np.arange(16))))
plt.plot(trimLevelArray, meanAverageArray, 'o')
plt.plot(trimLevelArray, poly(trimLevelArray))
plt.legend(['measured averages', f"fit: {round(poly(0))}x + {round(poly(1))}"])
plt.show()
