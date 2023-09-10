import metrics
from tabulate import tabulate
from abc import ABC, abstractmethod
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dataset import FairnessDataModule, SamplingTechnique
from training_module import TrainingModule, DistilledTrainingModule
import torch.nn as nn
from fairness import FairnessConstraint
from typing import Optional, Literal
import pytorch_lightning as pl
import utils


class BaseExperiment(ABC):
    """
    abstract base class that defines a blueprint for conducting
    fairness experiments using the PyTorch Lightning framework
    """
    def __init__(
            self,
            model_name: str,
            data_name: str,
            sensitive_attributes: list[str],
            lr: float = 0.0001,
            batch_size: int = 64,
            max_epochs: int = 50,
            patience: int = 9,
            scheduler_patience: int = 3,
            num_workers: int = 8,
            num_classes: int = 2,
            pretrained: bool = True,
            save_path: str = 'tmp/',
            random_seed: int = 42,
            sampling_technique: SamplingTechnique = SamplingTechnique.NO_SAMPLING,
            load_model_on_lr_change: bool = True,
            fairness_loss: FairnessConstraint = FairnessConstraint.NO_CONSTRAINT,
            lagrange: float = 0.0,
            epsilon: float = 0.0,
            skewed_data: bool = False,
            csv_logger: bool = True,
            wandb_logger: bool = False,
            gpu: int = 0,
        ):
                
        # Save parameters as instance variables
        self.model_name = model_name
        self.data_name = data_name
        self.sensitive_attributes = sensitive_attributes
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.scheduler_patience = scheduler_patience
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.save_path = save_path
        self.random_seed = random_seed
        self.sampling_technique = sampling_technique
        self.load_model_on_lr_change = load_model_on_lr_change
        self.csv_logger = csv_logger
        self.wandb_logger = wandb_logger
        self.gpu = gpu
        self.fairness_loss = fairness_loss
        self.lagrange = lagrange
        self.epsilon = epsilon
        self.skewed_data = skewed_data

        # TODO: CHANGE TO DYNAMIC DISTILLATION NOT NO_DISTILLATION
        self.model_id = utils.generate_model_id(
            model_name = self.model_name,
            random_seed = self.random_seed,
            sampling_technique = self.sampling_technique,
            distillation_technique = utils.DistillationTechnique.NO_DISTILLATION,
            fairness_loss = self.fairness_loss,
            lagrange = self.lagrange,
            epsilon = self.epsilon,
        )

        # Set seed
        pl.seed_everything(self.random_seed, workers=True)


    def create_data_module(self):
        return FairnessDataModule(
            sensitive_attributes=self.sensitive_attributes,
            dataset_name=self.data_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampling_technique=self.sampling_technique,
        )
    
    def create_loggers(self, log_name: str):
        loggers = []
        if self.csv_logger:
            loggers.append(CSVLogger(self.save_path, name=log_name))
        if self.wandb_logger:
            # TODO: Think about that projects name in WandbLogger (add it to signature)
            loggers.append(WandbLogger(project="resnets", name=log_name, log_model=True, save_dir=self.save_path))
        return loggers
    
    def create_training_module(self, model: nn.Module, checkpoint_path: Optional[str] = None):
        return TrainingModule(
            model=model,
            lr=self.lr, 
            patience=self.patience, 
            scheduler_patience=self.scheduler_patience, 
            load_model_on_lr_change=self.load_model_on_lr_change,
            checkpoint_path=checkpoint_path,
            fairness_loss=self.fairness_loss,
            lagrange=self.lagrange,
            epsilon=self.epsilon,
            skewed_data=self.skewed_data,
        )
    
    def create_distill_module(self, studnt_model: nn.Module, teacher_model: nn.Module):
        return DistilledTrainingModule(
            student_model=studnt_model,
            teacher_model=teacher_model,
            lr=self.lr,
            patience=self.patience,
            scheduler_patience=self.scheduler_patience,
            load_model_on_lr_change=self.load_model_on_lr_change,
            skewed_data=self.skewed_data,
            temperature=self.temperature,
            alpha=self.alpha,
            distillation_technique=self.distillation_technique,
        )
    
    def create_trainer(self, loggers: list[pl.loggers], log_name: str):
        return pl.Trainer(
            max_epochs=self.max_epochs,
            logger=loggers,
            callbacks=[
                EarlyStopping(monitor='val/loss', patience=self.patience),
                ModelCheckpoint(monitor="val/loss", save_top_k=1, mode="min", filename=log_name, save_weights_only=True),
            ],
            deterministic=True,
            accelerator='gpu',
            devices=[self.gpu]
        )

    def train(self, trainer: pl.Trainer, training_module: pl.LightningModule, data: FairnessDataModule):
        trainer.fit(training_module, data)

    def test(self, trainer: pl.Trainer, training_module: pl.LightningDataModule, data: FairnessDataModule, cpkt_path: str = 'None'):
        trainer.test(training_module, data, ckpt_path=cpkt_path)

    def predict_and_save(
            self,
            trainer: pl.Trainer,
            training_module: pl.LightningDataModule,
            data: FairnessDataModule,
            ckpt_path: str = None,
            save_predictions: bool = True,
            model_config: dict = None,
            verbose: bool = True,
        ):
        data.setup(stage='predict')
        data_loaders = data.predict_dataloader()
        predictions_test, predictions_val = trainer.predict(training_module, data_loaders, ckpt_path=ckpt_path)

        if self.skewed_data: 
            predictions = [('test', predictions_test)]
        else:
            predictions = [('test', predictions_test), ('val', predictions_val)]

        for mode, prediction in predictions: ### SKEWED DATA PREDICTIONS
            _, scores, labels, subgroups, loss, primary_loss, fair_loss = utils.format_predictions(prediction)
            accuracy_metrics = metrics.compute_overall_metrics(scores, labels)
            fairness_metrics = metrics.compute_fairness_metrics(scores, labels, subgroups)
            overall_metrics = accuracy_metrics | fairness_metrics
            overall_metrics['loss'] = loss
            overall_metrics['primary_loss'] = primary_loss
            overall_metrics['fair_loss'] = fair_loss

            # Pretty print the metrics for Test and Val
            if verbose:
                metric_rows = [[key, value] for key, value in overall_metrics.items()]
                print(f"\nMetrics for {mode}:")
                print(tabulate(metric_rows, headers=["Metric", "Value"], tablefmt="fancy_grid"))

            if save_predictions:
                 # Include information about the model and training config in the data_dict
                data_dict = model_config | overall_metrics
                save_path = self.save_path + trainer.logger.name + "/"
                utils.save_dicts_to_csv([data_dict], save_path + mode + ".csv") 
                utils.save_predictions_to_csv(prediction, save_path + mode + '_predictions' + ".csv")


    @abstractmethod
    def setup_training_components(self):
        raise NotImplementedError("Must override setup_training_components method in subclass")
    
    @abstractmethod
    def setup_testing_components(self):
        raise NotImplementedError("Must override setup_testing_components method in subclass")

    @abstractmethod
    def run_testing(self):
        raise NotImplementedError("Must override run_testing method in subclass")
    
    @abstractmethod
    def run_training(self):
        raise NotImplementedError("Must override run_training method in subclass")