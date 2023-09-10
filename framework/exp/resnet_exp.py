from exp.base_exp import BaseExperiment
from models import Resnet, FeatureResnet
import os
from dataset import SamplingTechnique
import utils
from tqdm import tqdm
from fairness import FairnessConstraint
import sys
import yaml
from training_module import TrainingModule
import pytorch_lightning as pl
import torch.nn as nn
from dataset import FairnessDataModule
import torch
from transforms import get_transforms_for_eval
import pandas as pd
from collections import OrderedDict



class ResnetExperiment(BaseExperiment):
    """
    This class inherits from the BaseExperiment class and provides a structured 
    framework to conduct experiments using ResNet models. 
    """
    def __init__(
            self, 
            model_name: str, 
            model_checkpoint: str = None,
            **kwargs
        ):
        super().__init__(model_name=model_name, **kwargs)
        self.model_checkpoint = model_checkpoint

        self.model = Resnet(model_name, self.num_classes, self.pretrained)

        print(self.model_id)

    def setup_training_components(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.data = super().create_data_module()
        loggers = super().create_loggers(self.model_id)
        self.trainer = super().create_trainer(loggers, self.model_id)
        self.training_module = super().create_training_module(self.model, self.model_checkpoint)
        self.model_config = utils.generate_model_config(
            model_name = self.model_name, 
            model_id = self.model_id,
            random_seed = self.random_seed, 
            sampling_technique = self.sampling_technique,
        )

        if self.fairness_loss is not FairnessConstraint.NO_CONSTRAINT: 
            self.model_config['fairness_loss'] = self.fairness_loss.value
            self.model_config['lagrange'] = self.lagrange
            self.model_config['epsilon'] = self.epsilon

    def setup_testing_components(self, save_predictions: bool = True):
        if not os.path.exists(self.save_path) and save_predictions:
            os.makedirs(self.save_path)
        self.data = super().create_data_module()
        loggers = super().create_loggers(self.model_id) if save_predictions else False
        self.trainer = super().create_trainer(loggers, self.model_id)
        self.training_module = super().create_training_module(self.model, self.model_checkpoint)
        self.model_config = utils.generate_model_config(
            model_name = self.model_name, 
            model_id = self.model_id,
            random_seed = self.random_seed, 
            sampling_technique = self.sampling_technique,
        )
        if self.fairness_loss is not FairnessConstraint.NO_CONSTRAINT: 
            self.model_config['fairness_loss'] = self.fairness_loss.value
            self.model_config['fairness_weight'] = self.fairness_weight

    def run_training(self, save_predictions: bool = True):
        self.setup_training_components()
        super().train(self.trainer, self.training_module, self.data)
        super().predict_and_save(
            trainer=self.trainer,
            training_module=self.training_module,
            data=self.data,
            ckpt_path='best',
            save_predictions=save_predictions,
            model_config=self.model_config,
        )

    def run_testing(self, save_predictions: bool = True, verbose: bool = True):
        self.setup_testing_components(save_predictions)
        super().predict_and_save(
            trainer=self.trainer, 
            training_module=self.training_module, 
            data=self.data, 
            ckpt_path=None,
            save_predictions=save_predictions,
            model_config=self.model_config,
            verbose=verbose
        )


# Main training Loop for ResNets called with config files
def train_resnets(
        data_names: list[str],
        sensitive_attributes: list[str],
        sampling_techniques: list[str] =["RANDOM_WEIGHTED_SAMPLING"],
        fairness_constraints: list[str] = ["NO_CONSTRAINT"],
        random_seeds = [42, 43, 44],
        model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101'],
        lagrangians = [0.0],
        epsilons = [0.00],
        num_workers: int = 8,
        skewed_data: bool = False,
        gpu: int = 0,
        save_path: str = None
    ):

    # Change sampling techniques to enum
    sampling_techniques = [utils.SamplingTechnique[sampling_technique] for sampling_technique in sampling_techniques]

    # Change fairness constraints to enum
    fairness_constraints = [utils.FairnessConstraint[fairness_constraint] for fairness_constraint in fairness_constraints]

    for data_name in data_names:
        save_dir = '../results/{data_name}/resnet/' if save_path is None else save_path
        for sampling_technique in sampling_techniques:
            for model in model_names:
                for fairness_constraint in fairness_constraints:
                    for lagrange in lagrangians:
                        for epsilon in epsilons:
                            for random_seed in random_seeds:
                                resnet_exp = ResnetExperiment(
                                    model_name=model,
                                    data_name=data_name,
                                    skewed_data=skewed_data,
                                    sensitive_attributes=sensitive_attributes,
                                    save_path=save_dir,
                                    gpu=gpu,
                                    num_workers=num_workers,
                                    sampling_technique=sampling_technique,
                                    random_seed=random_seed,
                                    fairness_loss=fairness_constraint,
                                    lagrange=lagrange,
                                    epsilon=epsilon,
                                )
                                resnet_exp.run_training(save_predictions=True)




if __name__ == "__main__":

    # Specify the experiment (correct config file) in the command line

    # Run with:
    # python3 -m exp.resnet_exp <config_file_name>

    experiment = sys.argv[1]
    with open(f'exp/resnet_configs/{experiment}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_resnets(**config)

    ### DEBUGGING
    # resnet_exp = ResnetExperiment(
    #     model_name='resnet18',
    #     data_name='ham10000',
    #     sensitive_attributes=['Age', 'Sex'],
    #     sampling_technique=SamplingTechnique.RANDOM_WEIGHTED_SAMPLING,
    #     random_seed=44,
    #     gpu=0,
    #     num_workers=8,
    #     max_epochs=5,
    # )
    # resnet_exp.run_training(save_predictions=False)