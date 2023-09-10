    
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch
import torch.nn.functional as F
from typing import Literal
from enum import Enum
import numpy as np

class FairnessConstraint(Enum):
    """Fairness Constraints used and their ids"""
    NO_CONSTRAINT = 0
    EQUAL_LOSS = 1
    EQUALIZED_ODDS = 2
    DISPARATE_IMPACT = 3
    EQUAL_AUC = 4


def get_fairness_loss(
        fairness_loss_name: FairnessConstraint,
        logits,
        labels,
        subgroup,
):
    # print(fairness_loss_name)
    if fairness_loss_name == FairnessConstraint.EQUAL_LOSS:
        return compute_equal_loss_constraint(logits, labels, subgroup)
    elif fairness_loss_name == FairnessConstraint.EQUALIZED_ODDS:
        return compute_equalized_odds_constraint(logits, labels, subgroup)
    elif fairness_loss_name == FairnessConstraint.DISPARATE_IMPACT:
        return compute_disparate_impact_constraint(logits, subgroup)
    elif fairness_loss_name == FairnessConstraint.NO_CONSTRAINT:
        return 0
    else:
        raise ValueError(f'Fairness loss {fairness_loss_name} not implemented.')


def compute_equalized_odds_constraint(logits, labels, subgroup):
    eq_odds = []
    scores = torch.softmax(logits, dim=1)
    preds = torch.argmax(scores, dim=1)
    for subgroup_values in subgroup.values():

        #### Equation from the Papers ####
        # fpr_0 = (preds * (1 - labels) * subgroup_values).sum() / subgroup_values.sum()
        # fpr_1 = (preds * (1 - labels) * (1 - subgroup_values)).sum() / (1 - subgroup_values).sum()
        # fnr_0 = ((1 - preds) * labels * subgroup_values).sum() / subgroup_values.sum()
        # fnr_1 = ((1 - preds) * labels * (1 - subgroup_values)).sum() / (1 - subgroup_values).sum()
        # eq_odds.append(abs(fpr_0 - fpr_1) + abs(fnr_0 - fnr_1))

    # return (max(eq_odds))

        #### Normal FPR and FNR from Confusion Matrix ####
        # Calculate FPR
        group_0_indices = subgroup_values.nonzero(as_tuple=True)
        group_0_labels = labels[group_0_indices]
        group_0_preds = preds[group_0_indices]
        cm = confusion_matrix(group_0_labels.cpu().detach().numpy(), group_0_preds.cpu().detach().numpy(), labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        fpr_group_0 = 0 if fp + tn == 0 else fp / (fp + tn)
        fnr_group_0 = 0 if fn + tp == 0 else fn / (fn + tp)
        
        # Calculate FNR 
        group_1_indices = (~subgroup_values).nonzero(as_tuple=True)
        group_1_labels = labels[group_1_indices]
        group_1_preds = preds[group_1_indices]
        cm = confusion_matrix(group_1_labels.cpu().detach().numpy(), group_1_preds.cpu().detach().numpy(), labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        fpr_group_1 = 0 if fp + tn == 0 else fp / (fp + tn)
        fnr_group_1 = 0 if fn + tp == 0 else fn / (fn + tp)

        eq_odds.append(abs(fpr_group_0 - fpr_group_1) + abs(fnr_group_0 - fnr_group_1))
    
    eq_odds = np.array(eq_odds)
    eq_odds = torch.from_numpy(eq_odds)
    return (max(eq_odds))


def compute_equal_auc_constraint(logits, labels, subgroup):
    auc_diffs = []
    scores = torch.softmax(logits, dim=1)
    for subgroup_values in subgroup.values():
        #### AUC for each subgroup ####
        group_0_indices = subgroup_values.nonzero(as_tuple=True)
        auroc_group_0 = roc_auc_score(
            y_true=labels[group_0_indices].cpu().detach().numpy(), y_score=scores[group_0_indices].cpu().detach().numpy()[:,1]
        )
        auroc_group_1 = roc_auc_score(
            y_true=labels[~group_0_indices].cpu().detach().numpy(), y_score=scores[~group_0_indices].cpu().detach().numpy()[:,1]
        )
        auc_diffs.append(abs(auroc_group_0 - auroc_group_1))
    return (max(auc_diffs))


def compute_equal_loss_constraint(logits, labels, subgroup):
    scores = torch.softmax(logits, dim=1)
    preds = torch.argmax(scores, dim=1)
    loss_diffs = []
    for subgroup_values in subgroup.values():
        # Calculate loss for group 0
        group_0_indices = subgroup_values.nonzero(as_tuple=True)
        group_0_labels = labels[group_0_indices]
        group_0_logits = logits[group_0_indices]
        loss_group_0 = F.cross_entropy(group_0_logits, group_0_labels)

        # Calculate loss for group 1
        group_1_indices = (~subgroup_values).nonzero(as_tuple=True)
        group_1_labels = labels[group_1_indices]
        group_1_logits = logits[group_1_indices]
        loss_group_1 = F.cross_entropy(group_1_logits, group_1_labels)
        
        # Add the differences to the lists
        loss_diffs.append(abs(loss_group_0 - loss_group_1))

    return (max(loss_diffs))

# TODO: Sth wrong with this metric -> Don't use it for now
def compute_disparate_impact_constraint(logits, subgroup):
    scores = torch.softmax(logits, dim=1)
    preds = torch.argmax(scores, dim=1)
    ratios = []
    for subgroup_values in subgroup.values():
        ratio_0 = ((preds * subgroup_values).sum() / subgroup_values.sum()) / ((preds * (1 - subgroup_values)).sum() / (1 - subgroup_values).sum())
        ratio_1 = ((preds * (1 - subgroup_values)).sum() / (1 - subgroup_values).sum()) / ((preds * subgroup_values).sum() / subgroup_values.sum())
        ratios.append(min(ratio_0, ratio_1))
    return -min(ratios)
