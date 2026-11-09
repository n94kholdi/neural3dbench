# Neural3DBench dashboard

Streamlit app that ties together the advisor, a 3D data preview, background MLflow
training and post-training inspection.

## Install

```bash
pip install -e ".[dashboard]"
```

## Launch

Run both commands from the repository root. Runs are written to `./mlruns` and weights
to `./checkpoints` relative to the working directory.

```bash
streamlit run tools/dashboard/app.py                     # dashboard on http://localhost:8501
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns --port 5000   # http://localhost:5000
```

MLflow 3 defaults `mlflow ui` to a SQLite store and rejects the file store unless
`MLFLOW_ALLOW_FILE_STORE=true` is set, so keep both parts to see the dashboard's runs.

## Tabs

1. **Problem & Advisor**: task, representation, points per sample and GPU memory give ranked
   recommendations (`AdvisorBridge.get_recommendations`). Each card opens the model dossier;
   The questions under *Describe the problem* change with the task and set the output the model
   must give (`required_output`). Each card cites the model's paper and its reported results.
   *Use for training* fills tab 3 from the paper's recipe (`AdvisorBridge.training_setup`): optimizer,
   learning rate, schedule, epochs, batch capped by GPU memory, rotation augmentation when rotations
   are arbitrary. It also picks a dataset that gives the required output (`tasks.choose_dataset`,
   preferring a downloaded paper benchmark). Tab 3 explains every choice under *Why this training setup*.
   When no registered model can do the task (instance segmentation,
   generation, scene flow), the advisor returns `verdict: "no_fit"`. The tab then shows
   *Not supported yet*, what the task needs and networks to add (with papers), and disables *Use for training*. *Estimate VRAM* gives the memory heuristic for any model.
   *Physics & data assumptions* (boundary conditions, target kind, required symmetries, topology/geometry
   changes, unseen resolution, GPU, dataset size, epochs) feed a **pre-flight check** on every card: the
   advisor's `audit_problem_assumptions`, `check_symmetry_and_invariance` and `estimate_training_budget`,
   shown as a passed / warnings / errors badge and an estimated training time. Cards also show the best
   measured result of the model from tab 5.
2. **Data**
   - *Preview geometry*: upload `.obj`, ASCII `.ply` (a per-point `label` property colors by class) or `.pt`
     (a `[N, 3]` tensor or a dict with `vertices`/`coords`, optional `faces` and `field`/`target`), or load a repository sample.
   - *Download a dataset*: ShapeNetPart (one category or all 16, part segmentation, 707 MB),
     ModelNet10 (473 MB) or ModelNet40 (2.0 GB) (classification, points sampled on the meshes).
     Archives are size- and (ShapeNetPart) sha256-checked and read without extracting.
   - *Upload a labelled dataset*: one `.npz` with `coords [B, N, 3]` and `labels` (`[B, N]` part ids or
     `[B]` class ids) or a float `target [B, N, C]`, optional `class_names`; or one labelled ASCII `.ply` per shape.
   - Saved datasets get a class table to rename classes and set hex colors, used by every preview.

   Everything is cached in `~/.cache/neural3dbench` (`NEURAL3DBENCH_DATA` overrides it) and
   registered again on start.
3. **Training**: pick a registered model and hyperparameters, then start or stop a background run.
   The dataset list only shows datasets the model can train on that give the problem's output
   (*Show all datasets* overrides). *Advanced* adds SGD momentum, weight decay, an LR schedule
   (`none`, `cosine:<eta_min>`, `step:<every>:<gamma>`, `multistep:<e1>,<e2>:<gamma>`) and random
   rotation augmentation. Loss curves and epoch progress refresh every 2 seconds. Each synthetic dataset names its task,
   and the task sizes the model head (`output_dim`, `task`) and picks the loss and metrics:

   | dataset | task | target | loss | metrics |
   |---|---|---|---|---|
   | `deformed_sphere`, `poisson` | field regression | float `[B, N, 1]` | MSE | MAE |
   | `chair_parts` | part segmentation (seat, back, legs, armrests) | int `[B, N]` | per-point CE | accuracy, mIoU |
   | `primitive_shapes` | shape classification (sphere, cube, cylinder) | int `[B]` | CE | accuracy |
   | `modelnet10_mesh`, `shrec11` | mesh classification (meshnet, diffusionnet, meshwalker only) | int `[B]` | CE | accuracy |
   | saved `.npz` datasets | from the arrays: `labels [B, N]`, `labels [B]` or `target` | | | |

   Mesh datasets are lists of `(vertices [V, 3], faces [F, 3])`, collated per batch with `collate_meshes`;
   they download on the first run (ModelNet10 473 MB, SHREC'11 3.2 MB). DiffusionNet operators are
   computed once per run. *Points per sample* is ignored for them.

   Warnings appear when the dataset's task differs from the advisor problem, and when the model needs
   graph or voxel input (gcn, gat, meshgraphnet, fno3d, unet3d, gino, deep_nurbs); those cannot start.
4. **Inspection**: the task's preview on validation samples (field: truth, prediction and |error|;
   parts: true parts, predicted parts and misclassified points; classes: true vs predicted label), final
   MLflow metrics, and *Launch New Run with Modified Params*, which copies the run's settings back into tab 3.
5. **Results**: every finished MLflow run in `./mlruns`. A leaderboard of the best model per dataset, ranked by
   the task's primary metric (mIoU for part segmentation, accuracy for classification, MAE for field
   regression, Chamfer for completion), a per-model bar chart and run table for one dataset, overlaid
   learning curves, and a CSV export. The same report is the `get_benchmark_results` MCP tool
   (`tools/mcp_advisor/results.py`). Synthetic datasets are smoke tests, not published benchmarks.

One trainer is shared by every browser session of a server process, so only one run trains at a time.

## Tests

```bash
pytest tools/dashboard/
```
