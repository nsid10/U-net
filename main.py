import os
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import tensorflow as tf

from effnet import *

np.random.seed = 42

h, w, c = 584, 565, 3
