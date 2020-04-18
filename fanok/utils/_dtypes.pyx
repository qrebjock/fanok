import numpy as np
cimport numpy as np


FLOAT_NUMPY_TYPE_MAP = {
    4 : np.float32,
    8 : np.float64
}

NP_DOUBLE_D_TYPE = FLOAT_NUMPY_TYPE_MAP[sizeof(double)]
