"""Train GAT on the shared synthetic node-regression example.

The dataset and optimization loop live in ``train_gcn_synthetic`` so GCN and
GAT are compared through the same ModelInput/ModelOutput training pipeline.
"""

from train_gcn_synthetic import main


if __name__ == "__main__":
    main(default_model="gat")
