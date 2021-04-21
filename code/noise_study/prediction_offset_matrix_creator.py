# In this file we would like to study the behaviour per pixel.
# We want to make a prediction using trim 0 and F for a third trim level and see per pixel how well
# that prediction fits.

# First we fetch the noise mean matrix for trim 0 1 and a third trim level.
# Then we go want to be able to make a prediction per pixel.
import glob

import numpy
import numpy as np

PIXEL_MATRIX_WIDTH = 256
PIXEL_MATRIX_HEIGHT = 256


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


def get_noise_mean_prediction(dec_trim_array: list, trim_matrix_array: list, row: int, col: int,
                              trim_level: str) -> int:
    dec_trim_level = int(trim_level, 16)

    trim_data_array = list(map(lambda trimMatrix: trimMatrix[row][col], trim_matrix_array))

    poly = np.poly1d(np.polyfit(dec_trim_array, trim_data_array, 1))

    return round(poly(dec_trim_level))


def check_for_zero_values(trim_data_array: list, row: int, col: int) -> bool:
    for trim_data in trim_data_array:
        if trim_data[row][col] == 0:
            return True

    return False


def make_prediction_matrix(trims_to_predict_with: list, trim_to_predict: str) -> np.array:
    """
    This method makes an array with the distance to the predicted value per pixel.

    :param trim_to_predict: the trim to predict.
    :param trims_to_predict_with: the trims used to predict with.
    :return: A matrix with distances to predicted value.
    """
    trims_to_predict_with_matrices = list(map(lambda trim: get_trim_matrix(trim), trims_to_predict_with))
    dec_trim_array = list(map(lambda trim: int(trim, 16), trims_to_predict_with))

    result = np.zeros((PIXEL_MATRIX_WIDTH, PIXEL_MATRIX_HEIGHT))
    for i in range(PIXEL_MATRIX_WIDTH):
        for j in range(PIXEL_MATRIX_HEIGHT):
            if not check_for_zero_values(trims_to_predict_with_matrices, i, j):
                result[i][j] = get_noise_mean_prediction(dec_trim_array, trims_to_predict_with_matrices, i, j,
                                                         trim_to_predict)

    return result


def make_prediction_offset_matrix(trims_to_predict_with: list, trim_to_predict: str) -> (np.array, np.array):
    real_values_matrix = get_trim_matrix(trim_to_predict)
    predicted_matrix = make_prediction_matrix(trims_to_predict_with, trim_to_predict)

    result = np.zeros((PIXEL_MATRIX_WIDTH, PIXEL_MATRIX_HEIGHT))
    masked = np.zeros((PIXEL_MATRIX_WIDTH, PIXEL_MATRIX_HEIGHT))

    for i in range(PIXEL_MATRIX_WIDTH):
        for j in range(PIXEL_MATRIX_HEIGHT):
            if real_values_matrix[i][j] != 0 and predicted_matrix[i][j] != 0:
                result[i][j] = predicted_matrix[i][j] - real_values_matrix[i][j]
            else:
                masked[i][j] = 1

    return result, masked


prediction_offset_trim_1, masked_trim_1 = make_prediction_offset_matrix(['0', 'F'], '1')
prediction_offset_trim_3, masked_trim_3 = make_prediction_offset_matrix(['0', 'F'], '3')
prediction_offset_trim_d, masked_trim_d = make_prediction_offset_matrix(['0', 'F'], 'D')


def save_prediction_and_mask(prediction_matrix: np.ndarray, mask_matrix: np.ndarray, predicted_trim: str):
    numpy.savetxt(f"../../data/prediction_offset_matrices/Prediction_Offset_0F_to_{predicted_trim}.csv",
                  prediction_matrix,
                  delimiter=',', fmt='%i')
    numpy.savetxt(f"../../data/prediction_offset_matrices/Masked_0F_to_{predicted_trim}.csv", mask_matrix,
                  delimiter=',', fmt='%i')


# FIXME: The path below might not be correct if this file is moved to another directory.
save_prediction_and_mask(prediction_offset_trim_1, masked_trim_1, '1')
save_prediction_and_mask(prediction_offset_trim_3, masked_trim_3, '3')
save_prediction_and_mask(prediction_offset_trim_d, masked_trim_d, 'D')
