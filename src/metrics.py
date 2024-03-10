import numpy as np
import pandas as pd

from src.kaggle_metrics.kaggle_kl_div import score


def get_score(preds: np.ndarray, targets: np.ndarray) -> float:
    oof = pd.DataFrame(preds.copy())
    oof["id"] = np.arange(len(oof))

    true = pd.DataFrame(targets.copy())
    true["id"] = np.arange(len(true))

    cv = score(solution=true, submission=oof, row_id_column_name="id")
    return cv
