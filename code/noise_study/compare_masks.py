import glob

import offset_and_mask_handler as mask_handler
import numpy as np

zero_mean_mask_1 = mask_handler.get_mask_matrices('1', ['Zero_Mean'])[0]
zero_mean_mask_3 = mask_handler.get_mask_matrices('3', ['Zero_Mean'])[0]

