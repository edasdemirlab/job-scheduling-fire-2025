# import required packages

import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import mip_setup as mip_setup
import mip_solve as mip_solve
import openpyxl

mip_inputs = mip_setup.InputsSetup()
mip_solve.mathematical_model_solve(mip_inputs)

# first successful run !!! March 28, 2023 - 3:35 pm
# successful run after all bugs are fixed !! March 29, 2023 - 17:00
