# Neural3DBench roadmap

Neural3DBench already has a broad model zoo (19 registered models), a model registry,
an MCP architecture advisor and a training dashboard. The main gap, as the README
notes, is that common datasets, benchmark protocols and leaderboards are still missing,
and almost every script in `examples/` trains on synthetic data. The bottleneck is no
longer the number of models but trustworthy results on real data.

## Priority 1: become a real benchmark

1. **Standard dataset loaders, one per task.** Each loader handles download, caching,
   checksums and fixed splits.
   - Classification and part segmentation: ModelNet40, ShapeNetPart, ScanObjectNN.
   - Shape completion: the full PCN dataset (not only the airplane category).
   - Object detection: KITTI and SUN RGB-D (PointPillars and VoteNet are already implemented).
   - Registration: ModelNet40 with random rigid transforms (DCP protocol), then 3DMatch.
   - Physics: DeepMind MeshGraphNets data (cylinder_flow, deforming_plate), AirfRANS,
     ShapeNet-Car CFD (as used by the Transolver and GINO papers).
2. **A benchmark protocol per task.** Fix the split, metrics, seeds and compute budget.
   - Classification / segmentation: overall accuracy (OA), mIoU.
   - Shape completion: Chamfer-L1, F-score@1%.
   - Registration: relative rotation error (RRE), relative translation error (RTE).
   - Detection: mAP@0.25.
   - Physics surrogates: relative L2 error.
3. **One training entrypoint.** Replace the `train_*_synthetic.py` scripts with a single
   command such as `n3b train --model X --task Y --config cfg.yaml`, logging to the
   existing MLflow setup.
4. **Reproduce published numbers.** Add a "paper vs ours" table to each `docs/<model>.md`.
   This is what makes the repository credible; a leaderboard is not trusted without it.
5. **Leaderboard.** Generate it automatically from MLflow runs into `docs/leaderboard.md`
   and show it in the dashboard.

## Priority 2: fill coverage gaps

The advisor currently returns `no_fit` for several tasks. Suggested additions, in order:

- **Modern point backbones:** Point Transformer v3 and a sparse-convolution baseline
  (MinkowskiNet or spconv). Current state-of-the-art work compares against these.
- **Instance segmentation:** for example a Mask3D-style model or a PTv3 head.
- **Scene flow:** FlowNet3D and the self-supervised NSFP baseline.
- **Generation:** a point-cloud diffusion model (PVD or LION).
- **Neural fields and rendering:** Occupancy Networks, DeepSDF, NeRF, 3D Gaussian
  Splatting. This is a large part of 3D computer vision not yet covered.
- **Equivariant models:** Vector Neurons, EGNN, e3nn. The catalog already records
  symmetries; these models let the benchmark test them.

## Priority 3: research utilities

- **Robustness suite:** Gaussian noise, point dropout, SO(3) rotation, density change and
  occlusion. Report the metric drop per corruption, in the style of ModelNet40-C.
- **Efficiency profiling:** measure parameters, FLOPs, latency and peak memory, and feed
  the measured values to the advisor in place of its heuristic estimates.
- **Shared geometric ops:** move kNN, farthest point sampling and ball query into one
  `ops/` module with optional CUDA backends (pytorch3d or torch_cluster). Today
  `pointnet2` and `dgcnn` each keep their own copies.
- **Pretrained weights:** host them on the Hugging Face Hub and load them through
  `ModelRegistry.create(..., pretrained=True)`.

## Priority 4: usability for other people

- Publish a PyPI release.
- Build a documentation site (mkdocs-material can reuse `docs/` as is).
- Add one Colab notebook per task.
- Generate the README model table from the registry so it cannot drift.
- Add a CPU smoke test in CI that runs every registered model on one batch.

## Suggested order of work

Pick one task and take it end to end before widening. Registration is a good first
candidate because DCP is in progress:

1. Real dataset loader with fixed splits.
2. Benchmark protocol and metrics.
3. Three models trained with the shared entrypoint.
4. Results that match the published numbers.
5. Leaderboard entry.

Then repeat the same template for the other tasks.
