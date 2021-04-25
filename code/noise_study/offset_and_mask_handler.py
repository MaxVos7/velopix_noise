import glob
import numpy as np

BASE_PATH = "../../data/prediction_offset_matrices/"


def make_file_path(trim: str, mask_name: str = '', for_mask: bool = True, prediction_base: str = '0F'):
    return f"{BASE_PATH}{'Mask_' + mask_name if for_mask else 'Prediction_Offset'}_{prediction_base}_to_{trim}.csv"


def fetch_prediction_offset_matrix(trim: str, prediction_base: str = '0F') -> np.ndarray:
    files = glob.glob(make_file_path(trim, for_mask=False, prediction_base=prediction_base))
    if len(files) < 1:
        raise FileNotFoundError(f"There is no prediction offset matrix for trim {trim}.")

    prediction_matrix = np.loadtxt(files[0], delimiter=',')

    return prediction_matrix


def make_file_from_matrix(matrix: np.ndarray,
                          trim: str,
                          mask_name: str = "",
                          prediction_base: str = "0F",
                          for_mask: bool = True):
    np.savetxt(
        make_file_path(trim, mask_name=mask_name, prediction_base=prediction_base, for_mask=for_mask),
        matrix,
        delimiter=',',
        fmt='%i'
    )


def get_mask_matrices(trim: str, mask_types=None, prediction_base: str = '0F'):
    if mask_types is None:
        files = glob.glob(make_file_path(trim, mask_name='*', prediction_base=prediction_base))
    else:
        files = fetch_masked_files_for_types(mask_types, trim, prediction_base=prediction_base)

    if len(files) < 1:
        print(f"There is no mask matrices found for trim {trim} (with the specified types).")
    elif mask_types is None:
        print("Found the following mask files:", files)

    mask_matrices = []

    for file in files:
        mask_matrices.append(np.loadtxt(file, delimiter=','))
    return mask_matrices


def fetch_masked_files_for_types(mask_types: list, trim: str, prediction_base: str = '0F') -> [str]:
    files = []
    for mask_type in mask_types:
        mask_files = glob.glob(make_file_path(trim, mask_name=mask_type, prediction_base=prediction_base))
        if len(mask_files) == 0:
            print(f"No mask files found for type: {mask_type}, skipping type.")
        elif len(mask_files) > 1:
            print(f"To many mask files found for type: {mask_type}, skipping type.")
        else:
            files.append(mask_files[0])
    return files


def is_masked(row: int, col: int, mask_matrices: list) -> bool:
    for mask_matrix in mask_matrices:
        if mask_matrix[row][col] == 1:
            return True

    return False


def filter_masked_pixels(matrix: np.ndarray, mask_matrices=None, trim=None, mask_types=None,
                         prediction_base: str = '0F') -> list:
    if mask_matrices is None and mask_types is None:
        print("filter_masked_pixels: No masks are specified.")
        return matrix.tolist()
    elif mask_matrices is None and mask_types is not None:
        if trim is None:
            raise AttributeError("When specifying mask_types, you must also give the trim level.")
        mask_matrices = get_mask_matrices(trim, mask_types, prediction_base=prediction_base)

    result = []

    for i in np.arange(matrix.shape[0]):
        for j in np.arange(matrix.shape[1]):
            if not is_masked(i, j, mask_matrices):
                result.append((matrix[i][j]))

    return result