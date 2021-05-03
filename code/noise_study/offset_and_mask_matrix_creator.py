import glob
import numpy as np
import offset_and_mask_handler as mh

PIXEL_MATRIX_WIDTH = 256
PIXEL_MATRIX_HEIGHT = 256


def get_trim_matrix(trim_level: str, module: int = 1, vp: str = '0-0') -> np.array:
    """
    This method fetches the noise mean array for a specific trim level.
    For this it uses the "Module1_VP0-0_Trim*_Noise_mean.csv file.
    """
    files = glob.glob(f"../../data/Module{module}_VP{vp}_Trim{trim_level}_Noise_Mean.csv")
    if len(files) < 1:
        raise FileNotFoundError(f"No file found for Noise mean of trim level {trim_level}.")
    else:
        file = files[0]

    return np.array(np.loadtxt(file, delimiter=',', dtype=int))


def get_noise_mean_prediction(dec_trim_array: list, trim_matrix_array: list, row: int, col: int,
                              trim_level: str, prediction_degree: int) -> int:
    dec_trim_level = int(trim_level, 16)

    trim_data_array = list(map(lambda trimMatrix: trimMatrix[row][col], trim_matrix_array))

    poly = np.poly1d(np.polyfit(dec_trim_array, trim_data_array, prediction_degree))

    return round(poly(dec_trim_level))


def check_for_zero_values(trim_data_array: list, row: int, col: int) -> bool:
    for trim_data in trim_data_array:
        if trim_data[row][col] == 0:
            return True

    return False


def make_prediction_matrix(trims_to_predict_with: list, trim_to_predict: str, prediction_degree: int,
                           module: int = None, vp: str = None) -> np.array:
    """
    This method makes an array with the distance to the predicted value per pixel.

    :param trim_to_predict: the trim to predict.
    :param trims_to_predict_with: the trims used to predict with.
    :return: A matrix with distances to predicted value.
    """
    trims_to_predict_with_matrices = list(
        map(lambda trim: get_trim_matrix(trim, module=module, vp=vp), trims_to_predict_with))
    dec_trim_array = list(map(lambda trim: int(trim, 16), trims_to_predict_with))

    result = np.zeros((PIXEL_MATRIX_WIDTH, PIXEL_MATRIX_HEIGHT))
    for i in range(PIXEL_MATRIX_WIDTH):
        for j in range(PIXEL_MATRIX_HEIGHT):
            if not check_for_zero_values(trims_to_predict_with_matrices, i, j):
                result[i][j] = get_noise_mean_prediction(dec_trim_array, trims_to_predict_with_matrices, i, j,
                                                         trim_to_predict, prediction_degree)

    return result


def make_prediction_offset_matrix(trims_to_predict_with: list, trim_to_predict: str, prediction_degree: int,
                                  module: int = None, vp: str = None) -> (np.array, np.array):
    real_values_matrix = get_trim_matrix(trim_to_predict, module=module, vp=vp)
    predicted_matrix = make_prediction_matrix(trims_to_predict_with, trim_to_predict, prediction_degree, module=module,
                                              vp=vp)

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
prediction_degree = 1
module = 1
vp = '0-1'
trims_to_predict = list(map(lambda value: format(value, 'x').upper(), np.arange(1, 15)))

for trim in trims_to_predict:
    prediction_offset, dead_mask = make_prediction_offset_matrix(prediction_base, trim, prediction_degree,
                                                                 module=module,
                                                                 vp=vp)
    mh.save_matrix(prediction_offset, trim,
                   mask_name='Prediction_Offset',
                   for_mask=False,
                   prediction_base=''.join(prediction_base),
                   module=module,
                   vp=vp
                   )
    mh.save_matrix(dead_mask, trim,
                   mask_name='Dead',
                   prediction_base=''.join(prediction_base),
                   module=module,
                   vp=vp
                   )
