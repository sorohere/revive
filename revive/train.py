# flake8: noqa
import os.path as osp
import sys  # Add sys import
from basicsr.train import train_pipeline

# Calculate project root_path and add it to sys.path
root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
if root_path not in sys.path:
    sys.path.append(root_path)

import revive.archs
import revive.data
import revive.models

if __name__ == '__main__':
    # root_path is already defined and added to sys.path
    train_pipeline(root_path)
