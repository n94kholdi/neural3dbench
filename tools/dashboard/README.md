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
   *Use for training* preselects the model in tab 3. *Estimate VRAM* gives the memory heuristic for any model.
2. **3D Data Preview**: upload `.obj`, ASCII `.ply` or `.pt` (a `[N, 3]` tensor or a dict with
   `vertices`/`coords`, optional `faces` and `field`/`target`), or load a repository sample.
3. **Training**: pick a registered model and hyperparameters, then start or stop a background run.
   Loss curves and epoch progress refresh every 2 seconds. Synthetic datasets feed
   `coordinates [B, N, 3]` and regress `[B, N, 1]`, so use coordinate or point models
   (`mlp`, `siren`, `pointnet`, ...).
4. **Inspection**: ground truth, prediction and absolute error on validation samples, final
   MLflow metrics, and *Launch New Run with Modified Params*, which copies the run's settings back into tab 3.

One trainer is shared by every browser session of a server process, so only one run trains at a time.

## Tests

```bash
pytest tools/dashboard/
```
