# MLFlow

Experiment tracking using MLflow.

## Start the tracking UI

```bash
source .venv/bin/activate
mlflow ui \
	--host 0.0.0.0 \
	--port 5000 \
	--backend-store-uri sqlite:///mlflow.db \
	--default-artifact-root ./mlartifacts
```

Open http://localhost:5000 in a browser. The tracking metadata is stored in
`mlflow.db`, and run artifacts are stored in `mlartifacts/`

## Train and log a model

Place the dataset at `data/water_potability.csv`, then run:

```bash
source .venv/bin/activate
python data-model.py
```

The script tunes a Random Forest with 3-fold cross-validation, optimizing F1
score, and compares it with a `BalancedRandomForestClassifier`. It logs
accuracy, balanced accuracy, precision, recall, and F1 for the baseline, tuned,
and balanced models. The selected hyperparameters, confusion matrices,
feature-importance chart, and trained models are also logged to MLflow.
Visualizations are available under the run's Artifacts tab. Balanced accuracy
is the average recall across both classes, so it is more informative than
ordinary accuracy when the target classes are imbalanced.

To name a particular run, use `--run-name`:

```bash
python data-model.py --run-name "random-forest-baseline"
```

The name appears in the MLflow UI and can be used to identify the run. To use
another dataset location, pass `--data-path /path/to/water_potability.csv`.
