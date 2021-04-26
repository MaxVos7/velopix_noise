# This files created some matrix files, for prediction offset and several masks.

import glob
import numpy as np
import offset_and_mask_handler as mh

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

    prediction_offset = np.zeros((PIXEL_MATRIX_WIDTH, PIXEL_MATRIX_HEIGHT))
    masked = np.zeros((PIXEL_MATRIX_WIDTH, PIXEL_MATRIX_HEIGHT))

    for i in range(PIXEL_MATRIX_WIDTH):
        for j in range(PIXEL_MATRIX_HEIGHT):
            prediction_offset[i][j] = predicted_matrix[i][j] - real_values_matrix[i][j]
            if real_values_matrix[i][j] == 0 or predicted_matrix[i][j] == 0:
                masked[i][j] = 1

    return prediction_offset, masked


def make_mask_file_for_pixels_outside_std(matrix: np.ndarray, dead_mask: np.ndarray, mean: int, std: int,
                                          std_amount: int):
    (width, height) = matrix.shape

    masked = np.zeros(matrix.shape)

    for i in np.arange(width):
        for j in np.arange(height):
            if (matrix[i][j] > mean + std * std_amount or matrix[i][j] < mean - std * std_amount) \
                    and dead_mask[i][j] == 0:
                masked[i][j] = 1

    return masked


prediction_base = ['0', 'F']

for trim in ['1', '3', 'D']:
    prediction_offset, dead_mask = make_prediction_offset_matrix(prediction_base, trim)
    mh.make_file_from_matrix(prediction_offset, trim,
                             mask_name='Prediction_Offset',
                             for_mask=False,
                             prediction_base=''.join(prediction_base)
                             )
    mh.make_file_from_matrix(dead_mask, trim,
                             mask_name='Dead',
                             prediction_base=''.join(prediction_base)
                             )

    # filter for zero mean mask and creating mask for pixels outside 1 std.
    offset_array_without_zero_mean = mh.filter_masked_pixels(prediction_offset, [dead_mask])
    outside_std_mask = make_mask_file_for_pixels_outside_std(prediction_offset,
                                                             dead_mask, np.mean(offset_array_without_zero_mean),
                                                             np.std(offset_array_without_zero_mean),
                                                             std_amount=2
                                                             )

    mh.make_file_from_matrix(outside_std_mask, trim,
                             mask_name='Outside_2Std_Not_Dead',
                             prediction_base=''.join(prediction_base)
                             )
