import sys

from pt_char2 import text_layer
from pt_utils import main_retrain

ds_path, voc_path, model_path, = sys.argv[1:]
main_retrain(ds_path, voc_path, text_layer, model_path)
