from exp.base_exp import BaseExperiment
from models import Resnet
from typing import Optional
import utils
from dataset import SamplingTechnique
import sys
import yaml
import os

class KDExperiment(BaseExperiment):
    """
    This class inherits from the BaseExperiment class and provides a structured 
    framework to conduct KD experiments using ResNet models.
    """
    def __init__(
            self, 
            student_model_name: str, 
            teacher_model_name: str,
            teacher_path: str,
            teacher_id: str,
            pretrained_student: bool = True,
            pretrained_teacher: bool = True,
            temperature: float = 8.0,
            alpha: float = 0.2,
            student_path: Optional[str] = None,
            distillation_technique: utils.DistillationTechnique = utils.DistillationTechnique.VANILLA,
            feature_maps_at: list[bool] = [False, False, False, False],
            **kwargs
        ):
        super().__init__(
            model_name=student_model_name,
            pretrained=pretrained_teacher,
            **kwargs
        )
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_path = teacher_path
        self.student_path = student_path
        self.teacher_id = teacher_id
        self.teacher_model_name = teacher_model_name
        self.distillation_technique = distillation_technique

        # Create teacher and student models
        # self.teacher = Resnet(teacher_model_name, self.num_classes, self.pretrained)
        # self.student = Resnet(self.model_name, self.num_classes, pretrained_student)

        if distillation_technique == utils.DistillationTechnique.FEATURE:
            if True not in feature_maps_at:
                raise ValueError("At least one feature map must be used for feature distillation")

        self.teacher = Resnet(teacher_model_name, self.num_classes, self.pretrained, feature_maps_at=feature_maps_at)
        self.student = Resnet(self.model_name, self.num_classes, self.pretrained, student=True, teacher_hint_channels=self.teacher._infer_feature_dim(), feature_maps_at=feature_maps_at)

        self.model_id = utils.generate_model_id(
            model_name=self.model_name,
            random_seed=self.random_seed,
            sampling_technique=self.sampling_technique,
            distillation_technique=self.distillation_technique,
            teacher_id=self.teacher_id,
            temperature=self.temperature,
            alpha=self.alpha,
            feature_maps_at=feature_maps_at
        )
        print(self.model_id)

    def setup_training_components(self):
        self.data = super().create_data_module()
        loggers = super().create_loggers(self.model_id)
        self.trainer = super().create_trainer(loggers, self.model_id)
        self.teacher_module = super().create_training_module(self.teacher, self.teacher_path) 
        # self.student_module = super().create_training_module(self.student) 
        # TODO: Add student path
        self.student_module = super().create_training_module(self.student, self.student_path)
        self.distill_module = super().create_distill_module(self.student, self.teacher_module.model)
        # TODO: Maybe it does not make sense to create a standalone training module for the teacher. Maybe it's better ot just load it here
        
    def setup_testing_components(self):
        self.data = super().create_data_module()
        loggers = super().create_loggers(self.model_id)
        self.trainer = super().create_trainer(loggers, self.model_id)
        self.student_module = super().create_training_module(self.student, self.student_path)

    def run_training(self, save_predictions: bool = True):
        self.setup_training_components()
        super().train(self.trainer, self.distill_module, self.data)
        super().predict_and_save(
            trainer=self.trainer,
            training_module=self.student_module, # testing on training module because we want don't need the teacher
            data=self.data,
            ckpt_path="best",
            save_predictions=save_predictions,
            model_config=utils.generate_model_config(
                model_name = self.model_name, 
                model_id = self.model_id,
                random_seed = self.random_seed, 
                sampling_technique = self.sampling_technique,
                distillation_technique=self.distillation_technique,
                teacher_name=self.teacher_model_name,
                teacher_id=self.teacher_id,
            )
        )

    def run_testing(self, save_predictions: bool = True, verbose: bool = True):
        self.setup_testing_components()
        super().predict_and_save(
            trainer=self.trainer,
            training_module=self.student_module, # testing on training module because we want don't need the teacher
            data=self.data,
            ckpt_path=None,
            save_predictions=save_predictions,
            model_config=utils.generate_model_config(
                model_name = self.model_name, 
                model_id = self.model_id,
                random_seed = self.random_seed, 
                sampling_technique = self.sampling_technique,
                distillation_technique=self.distillation_technique,
                teacher_name=self.teacher_model_name,
                teacher_id=self.teacher_id,
            ),
            verbose=verbose
        )


def run_kd(
    data_names: list[str],
    sensitive_attributes: list[str],
    student_models: list[str],
    teacher_models: list[dict[str, str]], # tuple of teacher name and teacher id
    random_seeds: list[int],
    experiment_name: str,
    sampling_techniques: list[str] = ['RANDOM_WEIGHTED_SAMPLING'],
    skewed_data: bool = False,
    alphas: list[float] = [0.2],
    temperatures: list[float] = [8.0],
    num_workers: int = 8,
    gpu: int = 0,
    teacher_dir_path: str = '../teachers/',
    save_path: str = None,
    distillation_technique: str = 'VANILLA',
    feature_maps_at: list[bool] = [False, False, False, True],
):
    num_workers = os.cpu_count() // 2

    # Change the sampling technique str to the actual enum
    sampling_techniques = [utils.SamplingTechnique[sampling_technique] for sampling_technique in sampling_techniques]

    # Change the distillation technique str to the actual enum
    distillation_technique = utils.DistillationTechnique[distillation_technique]
    
    for data_name in data_names:
        save_dir = f'../results/{data_name}/{experiment_name}/' if save_path is None else save_path
        for alpha in alphas:
            for temperature in temperatures:
                for sampling_technique in sampling_techniques:
                    for model in student_models:
                        for teacher_dict in teacher_models:
                            teacher_name = teacher_dict['name']
                            teacher_id = teacher_dict['id']
                            for random_seed in random_seeds:
                                kd_exp = KDExperiment(
                                    data_name=data_name,
                                    skewed_data=skewed_data,
                                    sensitive_attributes=sensitive_attributes,
                                    student_model_name=model,
                                    teacher_model_name=teacher_name,
                                    teacher_id=teacher_id,
                                    teacher_path=teacher_dir_path + teacher_id + '.ckpt',
                                    sampling_technique=sampling_technique,
                                    random_seed=random_seed,
                                    save_path=save_dir,
                                    gpu=gpu,
                                    num_workers=num_workers,
                                    alpha=alpha,
                                    temperature=temperature,
                                    distillation_technique=distillation_technique,
                                    feature_maps_at=feature_maps_at,
                                )
                                kd_exp.run_training(save_predictions=True)



if __name__ == "__main__":

    # Specify the expeiment (correct config file) in the command line 
    # i.e. python3 -m exp.kd_exp chexpert_no_female

    experiment = sys.argv[1]
    with open(f'exp/kd_configs/{experiment}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    run_kd(**config)

    #### DEBUGGING
    # kd_exp = KDExperiment(
    #     data_name='ham10000',
    #     sensitive_attributes=['Age', 'Sex'],
    #     student_model_name='resnet18',
    #     teacher_model_name='resnet101',
    #     teacher_id='r101-sam1-s42',
    #     teacher_path='../teachers/ham10000/r101-sam1-s42.ckpt',
    #     save_path='../tmp/',
    #     random_seed=42,
    #     gpu=0,
    #     num_workers=os.cpu_count(),
    #     sampling_technique=SamplingTechnique.RANDOM_WEIGHTED_SAMPLING,
    #     max_epochs=5,
    #     distillation_technique=utils.DistillationTechnique.VANILLA,
    # )
    # kd_exp.run_training(save_predictions=False)



