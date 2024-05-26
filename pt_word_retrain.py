import sys

from pt_utils import *
from pt_word import text_layer

ds_path, voc_path, model_path, = sys.argv[1:]
main_retrain(ds_path, voc_path, text_layer, model_path)
