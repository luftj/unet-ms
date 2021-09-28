import os

from config import valid_map_ext
def is_valid_map(f):
    return (os.path.splitext(f)[-1] in valid_map_ext) and (not ("_mask" in f))

def is_valid_mask(f):
    return (os.path.splitext(f)[-1] in valid_map_ext) and ("_mask" in f)