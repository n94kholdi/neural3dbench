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
   *Use for training* preselects the model in tab 3 and the dataset that matches the problem
   (`tasks.dataset_for_problem`, e.g. part segmentation selects `chair_parts`).
   When no registered model can do the task (instance segmentation,
   generation, scene flow), the advisor returns `verdict: "no_fit"`. The tab then shows
   *Not supported yet*, what the task needs and networks to add (with papers), and disables *Use for training*. *Estimate VRAM* gives the memory heuristic for any model.
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
   Loss curves and epoch progress refresh every 2 seconds. Each synthetic dataset names its task,
   and the task sizes the model head (`output_dim`, `task`) and picks the loss and metrics:

   | dataset | task | target | loss | metrics |
   |---|---|---|---|---|
   | `deformed_sphere`, `poisson` | field regression | float `[B, N, 1]` | MSE | MAE |
   | `chair_parts` | part segmentation (seat, back, legs, armrests) | int `[B, N]` | per-point CE | accuracy, mIoU |
   | `primitive_shapes` | shape classification (sphere, cube, cylinder) | int `[B]` | CE | accuracy |
   | saved `.npz` datasets | from the arrays: `labels [B, N]`, `labels [B]` or `target` | | | |

   Warnings appear when the dataset's task differs from the advisor problem, and when the model needs
   graph or voxel input (gcn, gat, meshgraphnet, fno3d, unet3d, gino, deep_nurbs); those cannot start.
4. **Inspection**: the task's preview on validation samples (field: truth, prediction and |error|;
   parts: true parts, predicted parts and misclassified points; classes: true vs predicted label), final
   MLflow metrics, and *Launch New Run with Modified Params*, which copies the run's settings back into tab 3.

One trainer is shared by every browser session of a server process, so only one run trains at a time.

## Tests

```bash
pytest tools/dashboard/
```
