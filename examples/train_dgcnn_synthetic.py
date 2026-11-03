"""Train DGCNN on synthetic shape classification or radial-field segmentation.

Shares data, training loop, and flags with train_pointnet_synthetic.py, e.g.
`python examples/train_dgcnn_synthetic.py --task segmentation --epochs 20`.
"""

from __future__ import annotations

import sys

from train_pointnet_synthetic import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--model", "dgcnn"]
    main()
