
import csv
import torch
from collections import OrderedDict
from metrics import compute_overall_metrics, compute_fairness_metrics
import numpy as np
import csv
import pandas as pd
from dataset import SamplingTechnique
from enum import Enum
from typing import Optional
from fairness import FairnessConstraint

class DistillationTechnique(Enum):
    """Distillation techniques used and their ids"""
    NO_DISTILLATION = 0
    VANILLA = 1
    FEATURE = 2
    ATTENTION = 3

def format_predictions(predictions):
    labels = torch.cat([x["labels"] for x in predictions]).cpu().detach().numpy()
    attributes = OrderedDict()
    for key in predictions[0]['subgroup'].keys():
        attributes[key] = torch.cat([output['subgroup'][key] for output in predictions]).cpu().detach().numpy()
    logits = torch.cat([x["logits"] for x in predictions]).cpu().detach()
    scores = torch.softmax(logits, dim=1).squeeze().cpu().detach().numpy()

    loss = torch.stack([x["loss"] for x in predictions]).mean().item()
    primary_loss = torch.stack([x["primary_loss"] for x in predictions]).mean().item()
    fair_loss = torch.stack([x["fair_loss"] for x in predictions]).mean().item()

    # return logits, scores, labels, attributes, loss
    return logits, scores, labels, attributes, loss, primary_loss, fair_loss

def save_dicts_to_csv(data, filename):
    # Open the CSV file in write mode and write the header row
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
      
    # Loop through the list of dictionaries and write each dictionary to a new row in the CSV file
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        for row in data:
            writer.writerow(row)

def save_predictions_to_csv(predictions, filename):
    logits, scores, labels, subgroups, _, _, _ = format_predictions(predictions)
    # # Open a new CSV file in 'write' mode
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Logits', 'Scores', 'Labels'] + list(subgroups.keys()))

        for row in zip(logits.tolist(), scores.tolist(), labels, *subgroups.values()):
            writer.writerow(row)

def compute_metrics_from_csv(csv_path):
    _, scores, labels, _ = load_csv_with_sensitive_attributes(csv_path)
    return compute_overall_metrics(scores, labels)

def compute_fairness_metrics_from_csv(csv_path):
    _, scores, labels, sensitive_attributes = load_csv_with_sensitive_attributes(csv_path)
    return compute_fairness_metrics(scores, labels, sensitive_attributes)

def load_csv_with_sensitive_attributes(csv_path):
    data = pd.read_csv(csv_path)
    logits = np.array([eval(x) for x in data['Logits']])
    scores = np.array([eval(x) for x in data['Scores']])
    labels = np.array(data['Labels'].values)
    sensitive_attributes = {}
    for column in data.columns:
        if column not in ['Logits', 'Scores', 'Labels']:
            sensitive_attributes[column] = np.array(data[column].values)
    return logits, scores, labels, sensitive_attributes


def generate_model_id(
    model_name: str,
    random_seed: int,
    sampling_technique: SamplingTechnique,
    distillation_technique: Optional[DistillationTechnique] = DistillationTechnique.NO_DISTILLATION,
    teacher_id: Optional[str] = None,
    fairness_loss: Optional[FairnessConstraint] = FairnessConstraint.NO_CONSTRAINT,
    lagrange: Optional[float] = None,
    epsilon: Optional[float] = None,
    temperature: Optional[float] = None,
    alpha: Optional[float] = None,
    feature_maps_at: Optional[list[bool]] = None,
):
    res_scale = ''.join(filter(str.isdigit, model_name)) 
    if distillation_technique == DistillationTechnique.NO_DISTILLATION:
        # ID for resnet models:
        if fairness_loss == FairnessConstraint.NO_CONSTRAINT:
            # ID for non-fairness-constrained models:
            # r<res_scale>-sample<sampling_technique>-s<random_seed>
            return 'r'+res_scale+'-sam'+str(sampling_technique.value)+'-s'+str(random_seed)
        else:
            # ID for fairness-constrained models:
            return 'r'+res_scale+'-sam'+str(sampling_technique.value)+'-s'+str(random_seed)+'-f'+str(fairness_loss.value)+'-l'+str(lagrange)+'-e'+str(epsilon)  
    elif distillation_technique == DistillationTechnique.VANILLA:
        # ID for KD models:
        # r<res_scale>-sample<sampling_technique>-kd<distillation_technique>-s<random_seed>d_<teacher_id>
        return 'r'+res_scale+'-sam'+str(sampling_technique.value)+'-kd'+str(distillation_technique.value)+'-s'+str(random_seed)+'_'+teacher_id


        # ID for alpha-temp testing
        # return 't'+str(temperature)+'-a'+str(alpha)+'-s'+str(random_seed)
        # ID for alpha testing
        # return 'a'+str(alpha)+'-s'+str(random_seed)

    elif distillation_technique == DistillationTechnique.FEATURE or distillation_technique == DistillationTechnique.ATTENTION:
        str_list = ['1' if b else '0' for b in feature_maps_at]
        str_val = ''.join(str_list)
        return 'r'+res_scale+'-sam'+str(sampling_technique.value)+'-kd'+str(distillation_technique.value)+'-fm'+str_val+'-s'+str(random_seed)+'_'+teacher_id


def generate_model_config(
    model_name: str,
    model_id: str,
    random_seed: int,
    sampling_technique: SamplingTechnique,
    distillation_technique: DistillationTechnique = DistillationTechnique.NO_DISTILLATION,
    teacher_name: str = None,
    teacher_id: str = None,
):
    return {
        'model_name': model_name,
        'model_id': model_id,
        'teacher_name': teacher_name,
        'teacher_id': teacher_id,
        'seed': random_seed,
        'sampling_technique': sampling_technique.value,
        'distillation_technique': distillation_technique.value,
    }


def generate_fairness_config(
    fairness_loss: FairnessConstraint,
    fairness_weight: float,
):
    return {
        'fairness_loss': fairness_loss.value,
        'fairness_weight': fairness_weight,
    }