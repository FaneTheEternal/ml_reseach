import sys

from pt_utils import main_voc
from pt_word import text_layer

ds_path, voc_path = sys.argv[1:]
main_voc(ds_path, text_layer, voc_path)
