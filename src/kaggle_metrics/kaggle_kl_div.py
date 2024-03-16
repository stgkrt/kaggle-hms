from typing import Optional

import kaggle_metric_utilities
import numpy as np
import pandas as pd
import pandas.api.types


class ParticipantVisibleError(Exception):
    pass


def kl_divergence(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    epsilon: float,
    micro_average: bool,
    sample_weights: Optional[pd.Series],
):
    # Overwrite solution for convenience
    for col in solution.columns:
        # Prevent issue with populating int columns with floats
        if not pandas.api.types.is_float_dtype(solution[col]):
            solution[col] = solution[col].astype(float)

        # Clip both the min and max following Kaggle conventions
        # for related metrics like log loss
        # Clipping the max avoids cases where the loss would be infinite or undefined,
        # clipping the min prevents users from playing games
        # with the 20th decimal place of predictions.
        submission[col] = np.clip(submission[col], epsilon, 1 - epsilon)

        y_nonzero_indices = solution[col] != 0
        solution[col] = solution[col].astype(float)
        solution.loc[y_nonzero_indices, col] = solution.loc[
            y_nonzero_indices, col
        ] * np.log(
            solution.loc[y_nonzero_indices, col]
            / submission.loc[y_nonzero_indices, col]
        )
        # Set the loss equal to zero
        # where y_true equals zero following the scipy convention:
        # https://docs.scipy.org/doc/scipy/reference/
        # generated/scipy.special.rel_entr.html#scipy.special.rel_entr
        solution.loc[~y_nonzero_indices, col] = 0

    if micro_average:
        return np.average(solution.sum(axis=1), weights=sample_weights)
    else:
        return np.average(solution.mean())


def competition_score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    epsilon: float = 10**-15,
    micro_average: bool = True,
    sample_weights_column_name: Optional[str] = None,
) -> float:
    """The Kullback–Leibler divergence.
    The KL divergence is technically undefined/infinite where the target equals zero.

    This implementation always assigns those cases a score of zero;
    effectively removing them from consideration.
    The predictions in each row must add to one so any probability
    assigned to a case where y == 0 reduces
    another prediction where y > 0, so crucially there is an important indirect effect.

    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    solution: pd.DataFrame
    submission: pd.DataFrame
    epsilon: KL divergence is undefined for p=0 or p=1. If epsilon is not null,
    solution and submission probabilities are clipped to max(eps, min(1 - eps, p).
    row_id_column_name: str
    micro_average: bool. Row-wise average if True, column-wise average if False.

    """
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    sample_weights = None
    if sample_weights_column_name:
        if sample_weights_column_name not in solution.columns:
            raise ParticipantVisibleError(
                f"{sample_weights_column_name} not found in solution columns"
            )
        sample_weights = solution.pop(sample_weights_column_name)

    if sample_weights_column_name and not micro_average:
        raise ParticipantVisibleError(
            "Sample weights are only valid if `micro_average` is `True`"
        )

    for col in solution.columns:
        if col not in submission.columns:
            raise ParticipantVisibleError(f"Missing submission column {col}")

    kaggle_metric_utilities.verify_valid_probabilities(solution, "solution")
    kaggle_metric_utilities.verify_valid_probabilities(submission, "submission")

    return kaggle_metric_utilities.safe_call_score(
        kl_divergence,
        solution,
        submission,
        epsilon=epsilon,
        micro_average=micro_average,
        sample_weights=sample_weights,
    )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    output_dir = "/kaggle/working/"
    exp_name = "exp000_b4"
    model_name = "tf_efficientnet_b4"
    oof_basename = "tf_efficientnet_b4_oof_df_versionexp001_b4_stage1.csv"
    oof_df_path = f"{output_dir}/{exp_name}/{oof_basename}"
    oof_df = pd.read_csv(oof_df_path)
    oof_df = oof_df[oof_df["fold"] == 0].reset_index(drop=True)

    target_cols = [
        "eeg_id",
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    solution = oof_df[target_cols]
    pred_cols = [col for col in oof_df.columns if "pred_" in col]
    pred_cols = ["eeg_id"] + pred_cols
    submission = oof_df[pred_cols]
    # columnをpred_を除いたものにrename
    submission = submission.rename(columns=lambda x: x.replace("pred_", ""))

    print(competition_score(solution, submission, "eeg_id"))
