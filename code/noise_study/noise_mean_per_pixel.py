# In this file we would like to study the behaviour per pixel.
# We want to make a prediction using trim 0 and F for a third trim level and see per pixel how well
# that prediction fits.

# First we fetch the noise mean matrix for trim 0 1 and a third trim level.
# Then we go want to be able to make a prediction per pixel.
import glob
import numpy as np
from matplotlib import pyplot as plt


def get_trim_matrix(trim_level: str) -> np.array:
    """
    This method fetches the noise mean array for a specific trim level.
    For this it uses the "Module1_VP0-0_Trim*_Noise_mean.csv file.
    """
    files = glob.glob(f"../../data/Module1_VP0-0_Trim{trim_level}_Noise_Mean.csv")
    if len(files) < 1:
        raise FileNotFoundError(f"No file found for Noise mean of trim level {trim_level}.")
    else:
        file = files[0]

    return np.array(np.loadtxt(file, delimiter=',', dtype=int))


def get_noise_mean_prediction(dec_trim_array: list, trim_matrix_array: list, row: int, col: int, trim_level: str) -> int:
    dec_trim_level = int(trim_level, 16)

    trim_data_array = list(map(lambda trimMatrix: trimMatrix[row][col], trim_matrix_array))

    poly = np.poly1d(np.polyfit(dec_trim_array, trim_data_array, 1))

    return poly(dec_trim_level)


def check_for_zero_values(trim_data_array: list, row: int, col: int) -> bool:
    for trim_data in trim_data_array:
        if trim_data[row][col] == 0:
            return True

    return False


def make_distance_to_predicted_array(trims_to_predict_with: list, trim_to_predict: str) -> tuple[list, int]:
    trims_to_predict_with_matrices = list(map(lambda trim: get_trim_matrix(trim), trims_to_predict_with))
    trim_to_predict_matrix = get_trim_matrix(trim_to_predict)

    """
    This method makes an array with the distance to the predicted value per pixel.

    :param trimArray: Give an array of trim levels, to be used for prediction (in same order as trimDataArray)
    :param trimDataArray: Give for each trim level, the matrix with means per pixel.
    :param trimToPredictArray: Give the trim matrix to predict.
    :param toPredictTrim: Give the trim level that is the to be predicted (trim level of the trimToPredictArray)
    :return: An 1D array of distance to predicted value.
    """
    distance_to_predicted_array = []
    amount_not_used = 0
    dec_trim_array = list(map(lambda trim: int(trim, 16), trims_to_predict_with))
    for i in range(trim_to_predict_matrix.shape[0]):
        for j in range(trim_to_predict_matrix.shape[1]):
            if trim_to_predict_matrix[i][j] != 0 and not check_for_zero_values(trims_to_predict_with_matrices, i, j):
                noise_mean_prediction = get_noise_mean_prediction(dec_trim_array, trims_to_predict_with_matrices, i, j,
                                                                trim_to_predict)
                distance_to_predicted_array.append(noise_mean_prediction - trim_to_predict_matrix[i][j])
            else:
                amount_not_used += 1

    return distance_to_predicted_array, amount_not_used


def make_histogram(prediction_offset_array: list, trims_to_predict_with: list, trim_to_predict: str, amount_not_used: int):
    hist_data = np.histogram(prediction_offset_array,
                             np.arange(min(prediction_offset_array), max(prediction_offset_array), 1))
    logs = hist_data[0].astype(float)
    mean = round(np.mean(prediction_offset_array), 2)
    plt.semilogy(hist_data[1][:-1], logs, linestyle='-',
                 label=f"{','.join(trims_to_predict_with)} -> {trim_to_predict}, mean:{mean}")
    print(f"{','.join(trims_to_predict_with)} -> {trim_to_predict}")
    print(f"mean: {mean}")
    print(f"points used: {len(prediction_offset_array)}")
    print(f"points not used: {amount_not_used}")


plt.xlabel("Predicted value - real value (current)")
plt.ylabel("The amount of times the difference is found")

(prediction_offset, amount_not_used) = make_distance_to_predicted_array(['0', '3', 'D', 'F'], 'D')

make_histogram(prediction_offset, ['0', '3', 'D', 'F'], 'D', amount_not_used)

(prediction_offset, amount_not_used) = make_distance_to_predicted_array(['0', '3', 'F'], 'D')

make_histogram(prediction_offset, ['0', '3', 'F'], 'D', amount_not_used)

(prediction_offset, amount_not_used) = make_distance_to_predicted_array(['0', 'F'], 'D')

make_histogram(prediction_offset, ['0', 'F'], 'F', amount_not_used)

plt.legend(loc='upper right')

plt.show()
