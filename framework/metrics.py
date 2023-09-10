import numpy as np
from typing import Dict
import sklearn.metrics as skm


def compute_overall_metrics(
    scores: np.ndarray, 
    labels: np.ndarray
) -> Dict[str, float]:
    preds = np.argmax(scores, axis=1)
    assert preds.shape == labels.shape, f"{preds.shape} != {labels.shape}"
    target_names = ["negative_class", "positive_class"]
    report = skm.classification_report(
        y_true=labels, y_pred=preds, output_dict=True, target_names=target_names, zero_division=0,
    )
    return {
        "accuracy": report["accuracy"],
        "f1": report["positive_class"]["f1-score"], 
        "auc": skm.roc_auc_score(y_true=labels, y_score=scores[:,1]),
        "youden": skm.balanced_accuracy_score(y_true=labels, y_pred=preds, adjusted=True),
        "PPV": report["positive_class"]["precision"],
        "NPV": report["negative_class"]["precision"], 
        "TPR": report["positive_class"]["recall"],
        "TNR": report["negative_class"]["recall"],
        "FPR": 1 - report["negative_class"]["recall"],
        "FNR": 1 - report["positive_class"]["recall"],
    }


def compute_fairness_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    attributes: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute relevant fairness metrics for 2-subgroup binary classification tasks,
    where 1 is the positive label and 0 is the negative label.

    N.B. Do not use in multi-class classification tasks.
    """
    # Compute fairness metrics for each sensitive attribute
    result_dict = {}
    max_auroc, max_f1, max_accuracy = 0, 0, 0
    min_auroc, min_f1, min_accuracy = 1, 1, 1
    for attribute_name, attribute_values in attributes.items():
        preds = np.argmax(scores, axis=1)
        assert (
            preds.shape == labels.shape == attribute_values.shape
        ), f"{preds.shape} != {labels.shape} != {attribute_values.shape}"
        target_names = ["negative_class", "positive_class"]

        group_0_mask = attribute_values == 0

        if np.sum(labels[group_0_mask]) != 0:
            auroc_group_0 = skm.roc_auc_score(
                y_true=labels[group_0_mask], y_score=scores[group_0_mask][:,1]
            )
        else:
            auroc_group_0 = 0

        if np.sum(labels[~group_0_mask]) != 0:
            auroc_group_1 = skm.roc_auc_score(
                y_true=labels[~group_0_mask], y_score=scores[~group_0_mask][:,1]
            )
        else:
            auroc_group_1 = 0

        report_group_0 = skm.classification_report(
            y_pred=preds[group_0_mask],
            y_true=labels[group_0_mask],
            output_dict=True,
            target_names=target_names,
            zero_division=0,
        )
        report_group_1 = skm.classification_report(
            y_pred=preds[~group_0_mask],
            y_true=labels[~group_0_mask],
            output_dict=True,
            target_names=target_names,
            zero_division=0,
        )

        # Youden's
        youden_group_0 = skm.balanced_accuracy_score(
            y_true=labels[group_0_mask], y_pred=preds[group_0_mask], adjusted=True
        )

        youden_group_1 = skm.balanced_accuracy_score(
            y_true=labels[~group_0_mask], y_pred=preds[~group_0_mask], adjusted=True
        )

        # Prevalance -> Proportion of group that has the sensitive attribute attribute
        num_group_0 = np.sum(attribute_values == 0)
        prevalence_group_0 = num_group_0 / len(attribute_values)
        num_group_1 = np.sum(attribute_values == 1)
        prevalence_group_1 = num_group_1 / len(attribute_values)

        # Disease prevalance -> Proportion of sensitive attribute group that has the disease
        disease_prevalence_group_0 = np.sum(labels[group_0_mask]) / num_group_0
        disease_prevalence_group_1 = np.sum(labels[~group_0_mask]) / num_group_1

        positive_preds_group_0 = np.sum(preds[group_0_mask]) / num_group_0
        positive_preds_group_1 = np.sum(preds[~group_0_mask]) / num_group_1

        accuracy_group_0 = report_group_0["accuracy"]  # type: ignore
        accuracy_group_1 = report_group_1["accuracy"]  # type: ignore

        f1_group_0 = report_group_0["positive_class"]["f1-score"]  # type: ignore
        f1_group_1 = report_group_1["positive_class"]["f1-score"]  # type: ignore

        tpr_group_0 = report_group_0["positive_class"]["recall"]  # type: ignore
        tpr_group_1 = report_group_1["positive_class"]["recall"]  # type: ignore
        tnr_group_0 = report_group_0["negative_class"]["recall"]  # type: ignore
        tnr_group_1 = report_group_1["negative_class"]["recall"]  # type: ignore

        # Demographic Parity
        dp = abs(positive_preds_group_0 - positive_preds_group_1)

        # Equality of Odds
        # Take the max out of the true positive rate difference or the true negative rate difference
        tpr_diff = abs(tpr_group_0 - tpr_group_1)
        tnr_diff = abs(tnr_group_0 - tnr_group_1)
        eq_odds = max(tpr_diff, tnr_diff)

        # Equality of Opportunity
        tpr_disease_diff = abs(disease_prevalence_group_0 - disease_prevalence_group_1)
        eq_opportunity = max(tpr_disease_diff, tnr_diff)

        # Record max and min for F1, AUROC, and Accuracy
        max_auroc = max(max_auroc, auroc_group_0, auroc_group_1)
        max_f1 = max(max_f1, f1_group_0, f1_group_1)
        max_accuracy = max(max_accuracy, accuracy_group_0, accuracy_group_1)
        min_auroc = min(min_auroc, auroc_group_0, auroc_group_1)
        min_f1 = min(min_f1, f1_group_0, f1_group_1)
        min_accuracy = min(min_accuracy, accuracy_group_0, accuracy_group_1)
        
        attribute_name = attribute_name.lower()
        subgroup_dict = {
            attribute_name+"_accuracy_group_0": accuracy_group_0,
            attribute_name+"_accuracy_group_1": accuracy_group_1,
            attribute_name+"_f1_group_0": f1_group_0,
            attribute_name+"_f1_group_1": f1_group_1,
            attribute_name+"_auc_group_0": auroc_group_0,
            attribute_name+"_auc_group_1": auroc_group_1,
            attribute_name+"_prevalence_group_0": prevalence_group_0,
            attribute_name+"_prevalence_group_1": prevalence_group_1,
            attribute_name+"_disease_prevalence_group_0": disease_prevalence_group_0,
            attribute_name+"_disease_prevalence_group_1": disease_prevalence_group_1,
            attribute_name+"_tpr_group_0": tpr_group_0,
            attribute_name+"_tpr_group_1": tpr_group_1,
            attribute_name+"_tnr_group_0": tnr_group_0,
            attribute_name+"_tnr_group_1": tnr_group_1,
            attribute_name+"_tpr_diff": tpr_diff,
            attribute_name+"_youden_group_0": youden_group_0,
            attribute_name+"_youden_group_1": youden_group_1,
            attribute_name+"_dp": dp,
            attribute_name+"_eq_odds": eq_odds,
            attribute_name+"_eq_opportunity": eq_opportunity,
        }

        result_dict = result_dict | subgroup_dict

    # Compute fairness metrics for the entire dataset
    auc_gap, f1_gap, accuracy_gap = abs(max_auroc - min_auroc), abs(max_f1 - min_f1), abs(max_accuracy - min_accuracy)
    result_dict['auc_gap'], result_dict['f1_gap'], result_dict['accuracy_gap']  = auc_gap, f1_gap, accuracy_gap
    return result_dict