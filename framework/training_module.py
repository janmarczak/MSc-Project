
from typing import Any, Dict, List, Literal, Mapping, Tuple, Optional

import torch
import torch.nn as nn
from transforms import get_transforms_for_train, get_transforms_for_eval
from metrics import compute_overall_metrics, compute_fairness_metrics
import pytorch_lightning as pl
import torch.nn.functional as F
from collections import OrderedDict
import warnings
from fairness import get_fairness_loss, FairnessConstraint
# warnings.filterwarnings("ignore", ".*does not have many workers.*")
from torch.autograd import Variable
from utils import DistillationTechnique


class TrainingModule(pl.LightningModule):
    """
    TrainingModule class is an extension of PyTorch Lightning's LightningModule
    to facilitate easier and more streamlined training of neural network models.
    It allows training, validation, and testing within the PyTorch Lightning framework.
    """
    def __init__(
      self, 
      model: nn.Module,
      lr: float = 0.0001,
      patience: int = 6,
      scheduler_patience: int = 3,
      load_model_on_lr_change: bool = False,
      checkpoint_path: Optional[str] = None,
      fairness_loss: FairnessConstraint = FairnessConstraint.NO_CONSTRAINT,
      lagrange: float = 0.0,
      epsilon: float = 0.1,
      skewed_data: bool = False,
    ):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.lr = lr
        self.model = model
        self.patience = patience
        self.scheduler_patience = scheduler_patience
        self.load_model_on_lr_change = load_model_on_lr_change
        self.fairness_loss = fairness_loss
        self.skewed_data = skewed_data

        self.train_transform = get_transforms_for_train(augment = True)
        self.eval_transform = get_transforms_for_eval()

        # Langrange Optimization
        if fairness_loss == FairnessConstraint.NO_CONSTRAINT:
            self.automatic_optimization = True
        else:
            self.automatic_optimization = False
        self.lagrange = torch.nn.Parameter(torch.tensor(lagrange), requires_grad=True)
        self.epsilon = torch.tensor(epsilon, requires_grad=False)

        # Load checkpoint if provided
        if checkpoint_path is not None:
            print('loading from checkpoint')
            if torch.cuda.is_available():
                # checkpoint = torch.load(checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
            else:
                self.load_state_dict(torch.load(checkpoint_path,  map_location=torch.device('cpu'))["state_dict"])
            
            # This is to delete any teacher weights from the checkpoint so that it's compatible for testing.
            # Previous experiments saved teacher weights, hence this is needed for compatibility.
            new_state_dict = {key: value for key, value in checkpoint["state_dict"].items() if not key.startswith('teacher')}
            checkpoint["state_dict"] = new_state_dict
            self.load_state_dict(checkpoint["state_dict"], strict=False)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=self.scheduler_patience, verbose=True
        )

        if self.automatic_optimization:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "frequency": 1,
                }
            }
        else:
            optimizer_lagrange = torch.optim.Adam([self.lagrange], lr=0.01, maximize=True)
            return [
                {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val/loss",
                        "frequency": 1,
                    }
                },
                {"optimizer": optimizer_lagrange},
            ]


    def training_step(self, batch, batch_idx):
        del batch_idx
        return self._step(batch, self.train_transform, "train")

    def on_train_epoch_end(self):
        self._epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        if self.trainer.state.stage == "sanity_check":
            return
        del batch_idx
        return self._step(batch, self.eval_transform, "val")

    def on_validation_epoch_end(self):
        if self.trainer.state.stage == "sanity_check":
            return
        self._epoch_end(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        del batch_idx
        return self._step(batch, self.eval_transform, "test")

    def on_test_epoch_end(self):
        self._epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()  # free memory

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, self.eval_transform, "predict")


    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        transform: nn.Module,
        mode: Literal["train", "val", "test"],
    ) -> Mapping[str, torch.Tensor]:
        
        #### LOAD BEST MODEL IN CASE OF LR CHANGE ####
        if self.load_model_on_lr_change and (mode=='train' or mode=='val'):
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            if self.lr != current_lr:
                print("load best model")
                best_checkpoint_path = self.trainer.callbacks[-1].best_model_path
                self.load_state_dict(torch.load(best_checkpoint_path)["state_dict"])
                self.lr = current_lr


        imgs, labels, subgroup  = batch
        imgs = transform(imgs)
        # logits = self(imgs)
        _, _, logits = self.model(imgs)

        if self.automatic_optimization: 
            #### 1. NORMAL TRAINING: Cross Entropy Loss
            loss = F.cross_entropy(logits, labels)
            primary_loss = loss
            fair_loss = torch.tensor(0.0)
        else:
            # TODO: What happens when fairness loss is nan -> then fair loss 1 or maybe just ignore (it's batch size 1)
            #### 2. FAIRNESS TRAINING: Cross Entropy Loss + Lagrange Multiplier
            optimizer, lagrange_optimizer = self.optimizers()
            primary_loss = F.cross_entropy(logits, labels)
            fair_loss = get_fairness_loss(self.fairness_loss, logits, labels, subgroup)
            loss = primary_loss - self.lagrange * (self.epsilon - fair_loss)
            if mode == "train":
                optimizer.zero_grad()
                lagrange_optimizer.zero_grad()
                self.manual_backward(loss)
                lagrange_optimizer.step() # gradient ascent
                optimizer.step()

        result_dict = {
            "loss": loss,
            "primary_loss": primary_loss,
            "fair_loss": fair_loss,
            "logits": logits,
            "labels": labels,
            "subgroup": subgroup
        }

        if mode == "train":
            self.training_step_outputs.append(result_dict)
        elif mode == "val":
            self.validation_step_outputs.append(result_dict)
        elif mode == "test":
            self.test_step_outputs.append(result_dict)

        return result_dict


    def _epoch_end(
      self,
      step_outputs: List[Mapping[str, torch.Tensor]],
      mode: Literal["train", "val", "test"],
    ):

        labels = torch.cat([x["labels"] for x in step_outputs]).cpu().detach().numpy()
        attributes = OrderedDict()
        for key in step_outputs[0]['subgroup'].keys():
            attributes[key] = torch.cat([output['subgroup'][key] for output in step_outputs]).cpu().detach().numpy()
        logits = torch.cat([x["logits"] for x in step_outputs])
        scores = torch.softmax(logits, dim=1).squeeze().cpu().detach().numpy()

        accuracy_metrics_dict = compute_overall_metrics(scores=scores, labels=labels)

        #### When training with skewed data we don't compute fairness validation metrics (no sensitive attributes)
        if self.skewed_data:
            if mode == "test":
                fairness_metrics_dict = compute_fairness_metrics(scores, labels, attributes)
            else:
                fairness_metrics_dict = {}
        else:
            fairness_metrics_dict = compute_fairness_metrics(scores, labels, attributes)

        metrics_dict = accuracy_metrics_dict | fairness_metrics_dict

        # Not including NaN values (sometimes there are NaN values in the fair loss if we have a batch size of 1
        # metrics_dict["loss"] = torch.stack([x["loss"] for x in step_outputs]).mean().item()
        metrics_dict["loss"] = torch.mean(torch.masked_select(torch.stack([x["loss"] for x in step_outputs]), ~torch.isnan(torch.stack([x["loss"] for x in step_outputs])))).item()
        metrics_dict["primary_loss"] = torch.stack([x["primary_loss"] for x in step_outputs]).mean().item()
        metrics_dict['fair_loss'] = torch.mean(torch.masked_select(torch.stack([x["fair_loss"] for x in step_outputs]), ~torch.isnan(torch.stack([x["fair_loss"] for x in step_outputs])))).item()
        
        if mode == 'val' and self.automatic_optimization == False:
            sch = self.lr_schedulers()
            sch.step(metrics_dict["loss"])

        # Converting to dict and adding the mode
        torch_metrics_dict_with_mode = {
            f"{mode}/{k}": v for k, v in metrics_dict.items()
        }

        # Save the logs with all the metrics
        self.log_dict(torch_metrics_dict_with_mode, on_epoch=True, prog_bar=True, logger=True)


class DistilledTrainingModule(TrainingModule):
    """
    Extension for training models using Knowledge Distillation
    """
    def __init__(
            self, 
            student_model: nn.Module, 
            teacher_model: nn.Module, 
            lr: float = 0.0001,
            patience: int = 6,
            scheduler_patience: int = 3,
            load_model_on_lr_change: bool = False,
            temperature: int = 10, 
            alpha: float = 0.1,
            skewed_data: bool = False,
            distillation_technique: DistillationTechnique = DistillationTechnique.VANILLA,
            student_checkpoint_path: Optional[str] = None,
        ):
        
        super(DistilledTrainingModule, self).__init__(
            model=student_model, 
            checkpoint_path=student_checkpoint_path,
            lr = lr, 
            patience=patience,
            scheduler_patience=scheduler_patience,
            skewed_data=skewed_data,
        )

        self.teacher_model = teacher_model # TODO: Consider copying the model instead of passing it in
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.temperature = temperature
        self.alpha = alpha
        self.load_model_on_lr_change = load_model_on_lr_change
        self.distillation_technique = distillation_technique

        if self.distillation_technique == DistillationTechnique.VANILLA:
            self.distillation_loss_fn = nn.KLDivLoss(reduction='batchmean')
        elif self.distillation_technique == DistillationTechnique.FEATURE or self.distillation_technique == DistillationTechnique.ATTENTION:
            self.distillation_loss_fn = nn.MSELoss()
            self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        
    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        transform: nn.Module,
        mode: Literal["train", "val", "test"],
    ) -> Mapping[str, torch.Tensor]:
    
        #### LOAD BEST STUDENT MODEL IN CASE OF LR CHANGE ####
        if self.load_model_on_lr_change and (mode=='train' or mode=='val'):
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            if self.lr != current_lr:
                print("load best model")
                best_checkpoint_path = self.trainer.callbacks[-1].best_model_path
                loaded_state_dict = torch.load(best_checkpoint_path)["state_dict"]

                # Remove the "model." prefix from the keys -> Hard coded but needs to be done to load the student model
                new_state_dict = {}
                for key in loaded_state_dict:
                    if key.startswith("model."):
                        new_key = key.replace("model.", "")
                        new_state_dict[new_key] = loaded_state_dict[key]

                # Load the student model
                self.model.load_state_dict(new_state_dict)
                self.lr = current_lr

        imgs, labels, subgroup = batch
        imgs = transform(imgs)

        feature_s, attention_s, student_output = self.model(imgs)
        feature_t, attention_t, teacher_output = self.teacher_model(imgs)

        ### Calculate Loss
        # Student Loss
        student_loss = F.cross_entropy(student_output, labels)

        if self.distillation_technique == DistillationTechnique.VANILLA:
            # Scaled Distillation Loss from KERAS based on https://arxiv.org/abs/1503.02531 
            # https://keras.io/examples/vision/knowledge_distillation/
            # Other resource: https://github.com/haitongli/knowledge-distillation-pytorch/blob/master/model/net.py

            distillation_loss = (
                self.distillation_loss_fn(
                      F.log_softmax(student_output / self.temperature, dim=1),
                      F.softmax(teacher_output / self.temperature, dim=1)
                )
                * self.temperature**2
            )
    
        elif self.distillation_technique == DistillationTechnique.FEATURE:

            # Original Matching only feature maps
            if len(feature_s) != len(feature_t):
                raise ValueError("Student and Teacher feature maps lists are not of the same length")
            
            total_loss = 0.0
            for fmap_s, fmap_t in zip(feature_s, feature_t):
                total_loss += self.distillation_loss_fn(fmap_s, fmap_t)
            distillation_loss = total_loss / len(feature_s)

            kl_loss = (
                self.kl_loss_fn(
                      F.log_softmax(student_output / self.temperature, dim=1),
                      F.softmax(teacher_output / self.temperature, dim=1)
                )
                * self.temperature**2
            )

        elif self.distillation_technique == DistillationTechnique.ATTENTION:
            total_loss = 0.0
            for amap_s, amap_t in zip(attention_s, attention_t):
                if amap_s.shape[2:] != amap_t.shape[2:]:
                    amap_t = F.interpolate(amap_t, size=amap_s.shape[2:], mode='bilinear', align_corners=False)
                total_loss += self.distillation_loss_fn(amap_s, amap_t)
            distillation_loss = total_loss / len(attention_s)

            kl_loss = (
                self.kl_loss_fn(
                      F.log_softmax(student_output / self.temperature, dim=1),
                      F.softmax(teacher_output / self.temperature, dim=1)
                )
                * self.temperature**2
            )

        loss = 0.2 * student_loss + 0.6 * distillation_loss + 0.2 * kl_loss
        # loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        primary_loss = loss
        fair_loss = torch.tensor(0.0)

        # Save outputs
        result_dict = {
            "loss": loss,
            "primary_loss": primary_loss,
            "fair_loss": fair_loss,
            "logits": student_output,
            "teacher_logits": teacher_output,
            "labels": labels,
            "subgroup": subgroup,
        }

        if mode == "train":
            self.training_step_outputs.append(result_dict)
        elif mode == "val":
            self.validation_step_outputs.append(result_dict)
        elif mode == "test":
            self.test_step_outputs.append(result_dict)

        return result_dict
    

    def on_save_checkpoint(self, checkpoint):
        # Save only stuff that does not start with "teacher" (essentially saving only the student model weights)
        new_state_dict = {key: value for key, value in checkpoint["state_dict"].items() if not key.startswith('teacher')}
        checkpoint["state_dict"] = new_state_dict
        checkpoint