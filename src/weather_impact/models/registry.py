"""Save and load trained models with a JSON metadata sidecar.

Security note: joblib files can execute code when loaded. Only load model
files that this project created in models/saved/.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Union

import joblib
import sklearn
import xgboost
from sklearn.pipeline import Pipeline

from weather_impact.config.settings import (
    CATEGORICAL_FEATURES,
    MODEL_DIR,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from weather_impact.utils.exceptions import ModelNotFoundError

PathLike = Union[str, Path]


def _model_path(name: str, model_dir: PathLike) -> Path:
    return Path(model_dir) / f"{name}.joblib"


def _metadata_path(name: str, model_dir: PathLike) -> Path:
    return Path(model_dir) / f"{name}.json"


def save_model(
    pipeline: Pipeline,
    name: str,
    metrics: Dict[str, float],
    algorithm: str,
    model_dir: PathLike = MODEL_DIR,
) -> Path:
    """Persist a trained pipeline and its metadata. Returns the model file path."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = _model_path(name, model_dir)
    joblib.dump(pipeline, model_path)

    metadata = {
        "name": name,
        "algorithm": algorithm,
        "target": TARGET_COLUMN,
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    _metadata_path(name, model_dir).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path


def load_model(name: str, model_dir: PathLike = MODEL_DIR) -> Pipeline:
    """Load a trained pipeline. Raises ModelNotFoundError if it is missing."""
    path = _model_path(name, model_dir)
    if not path.exists():
        raise ModelNotFoundError(
            f"No trained model '{name}' found in {Path(model_dir)}. "
            "Train one first with: python scripts/train_model.py"
        )
    return joblib.load(path)


def load_metadata(name: str, model_dir: PathLike = MODEL_DIR) -> Dict[str, Any]:
    """Load the metadata sidecar for a saved model."""
    path = _metadata_path(name, model_dir)
    if not path.exists():
        raise ModelNotFoundError(f"No metadata found for model '{name}' in {Path(model_dir)}.")
    return json.loads(path.read_text(encoding="utf-8"))
