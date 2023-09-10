import os
from collections import OrderedDict
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from dataset import FairnessDataModule
from models import Resnet
from training_module import TrainingModule
from transforms import get_transforms_for_eval
import yaml
import sys
import re

"""
File for extracting feature embeddings from the 
penultimate layer of already trained ResNet models
"""

def create_embeddings_for_all(
    path: str,  # '../results/chexpert/resnet/'
    sensitive_attributes: list[str],
    data_name: str,
    num_workers: int = 4,
    gpu: int = 0,
    is_student: bool = False,
    is_kd: bool = False,
    teacher_id: str = None,
):
    
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    def find_ckpt_file(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".ckpt"):
                    return os.path.join(root, file)
        return None  # Return None if no .ckpt file is found
    
    try:
        folder_names = os.listdir(path)
        folder_names = [folder for folder in folder_names if 'sam1' in folder] # To only do Random Weighted Sampling
        if is_student:
            folder_names = [folder for folder in folder_names if (folder.startswith('r18') or folder.startswith('r34'))] # For students

        if is_kd:
            folder_names = [folder for folder in folder_names if teacher_id in folder] # filter the correct teacher


    except Exception as e:
        print(f"Error: {e}")

    print(folder_names)
    for folder_name in folder_names:
        print(folder_name)
        model_folder = os.path.join(path, folder_name)
        model_checkpoint = find_ckpt_file(model_folder)

        if is_kd:
            match = re.search(r'\br(\d+).*?\bs(\d+)', folder_name)
        else:
            match = re.search(r'r(\d+)-sam1-s(\d+)', folder_name)
        if match:
            scale = match.group(1)
            seed = match.group(2)
            model_name = f'resnet{scale}'

        # Set the seed
        pl.seed_everything(seed)
        
        # Load the model
        training_module = TrainingModule(model=Resnet(model_name, 2, True), checkpoint_path=model_checkpoint)
        model = training_module.model.to(device)

        # Create the data module
        data_module = FairnessDataModule(
            sensitive_attributes=sensitive_attributes,
            dataset_name=data_name,
            batch_size=64,
            num_workers=num_workers,
        )
        data_module.setup('test')
        test_loader = data_module.test_dataloader()

        # Traverse through data loader and get the embeddings
        embeddings = []
        targets = []
        subgroups = []
        model.eval()
        with torch.no_grad():
            for index, batch in enumerate(tqdm(test_loader, desc='Test-loop')):
                imgs, lab, sbgrps = batch
                imgs = imgs.to(device)
                imgs = get_transforms_for_eval()(imgs)
                embeds = model._backbone(imgs)

                # Store them
                embeddings.append(embeds)
                targets.append(lab)
                subgroups.append(sbgrps)

            embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            attributes = OrderedDict()
            for key in subgroups[0].keys():
                attributes[key] = torch.cat([d[key] for d in subgroups]).cpu().numpy()
            
        # Create a dataframe out of them and concatnate them
        # Save to a csv file
        df = pd.DataFrame(data=embeddings)
        df_targets = pd.DataFrame(data=targets, columns=['Targets'])
        df_subgroups = pd.DataFrame(data=attributes)
        df = pd.concat([df, df_targets, df_subgroups], axis=1)
        df.to_csv(f'{model_folder}/embeds.csv', index=False)


    

def create_embeddings(
    data_name: str,
    sensitive_attributes: list[str],
    models: list[dict[str, str]], # list of dicts with model_id, seed and model_path (maybe model name as well?)
    num_workers: int = 8,
    gpu: int = 0,
):
    
    def find_ckpt_file(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".ckpt"):
                    return os.path.join(root, file)
        return None  # Return None if no .ckpt file is found
    
    # Need to set up a GPU 'device' for the model to be on
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    
    for model_dict in models:
        random_seed = model_dict['random_seed']
        pl.seed_everything(random_seed)

        # Retrieve the model
        model_name = model_dict['name']
        model_id = model_dict['id']
        print(model_id)
        model_path = model_dict['path'] + model_id
        model_checkpoint = find_ckpt_file(model_path)

        # Load the model
        training_module = TrainingModule(model=Resnet(model_name, 2, True), checkpoint_path=model_checkpoint)
        model = training_module.model.to(device)

        # Create the data module
        data_module = FairnessDataModule(
            sensitive_attributes=sensitive_attributes,
            dataset_name=data_name,
            batch_size=64,
            num_workers=num_workers,
        )
        data_module.setup('test')
        test_loader = data_module.test_dataloader()

        # Traverse through data loader and get the embeddings
        embeddings = []
        targets = []
        subgroups = []
        model.eval()
        with torch.no_grad():
            for index, batch in enumerate(tqdm(test_loader, desc='Test-loop')):
                imgs, lab, sbgrps = batch
                imgs = imgs.to(device)
                imgs = get_transforms_for_eval()(imgs)
                embeds = model._backbone(imgs)

                # Store them
                embeddings.append(embeds)
                targets.append(lab)
                subgroups.append(sbgrps)

            embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            attributes = OrderedDict()
            for key in subgroups[0].keys():
                attributes[key] = torch.cat([d[key] for d in subgroups]).cpu().numpy()
            
        # Create a dataframe out of them and concatnate them
        # Save to a csv file
        df = pd.DataFrame(data=embeddings)
        df_targets = pd.DataFrame(data=targets, columns=['Targets'])
        df_subgroups = pd.DataFrame(data=attributes)
        df = pd.concat([df, df_targets, df_subgroups], axis=1)
        df.to_csv(f'{model_path}/embeds.csv', index=False)




if __name__ == "__main__":

    # Specify the experiment (correct config file) in the command line
    # To run ex: python3 -m exp.embeddings ham10000

    # experiment = sys.argv[1]
    # with open(f'exp/configs/{experiment}_embeddings.yaml', 'r') as f:
    #     config = yaml.safe_load(f)
    # create_embeddings(**config)

    import os
    workers = os.cpu_count()

    #### VANILLA-KD MODELS HIGH CAPACITY
    # 1. CHEXPERT

    # print('Vanilla-KD Models CheXpert')

    # create_embeddings_for_all(
    #     path='../results/chexpert_no_female/kd/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert_no_female',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r101-sam1-s46'
    # )

    # create_embeddings_for_all(
    #     path='../results/chexpert_no_old/kd/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert_no_old',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r101-sam1-s46'
    # )

    # create_embeddings_for_all(
    #     path='../results/chexpert_no_white/kd/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert_no_white',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r101-sam1-s46'
    # )

    # 2. HAM10000

    # print('Vanilla-KD Models HAM10000')

    # create_embeddings_for_all(
    #     path='../results/ham10000_no_female/resnet/',
    #     sensitive_attributes=['Age', 'Sex'],
    #     data_name='ham10000_no_female',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    # )

    # create_embeddings_for_all(
    #     path='../results/ham10000_no_old/resnet/',
    #     sensitive_attributes=['Age', 'Sex'],
    #     data_name='ham10000_no_old',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    # )

    

    # #### FEATURE-KD MODELS HIGH CAPACITY
    # # 1. CHEXPERT

    # print('Feature-KD Models CheXpert')

    # create_embeddings_for_all(
    #     path='../results/chexpert_no_female/kd-feature_rgb/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert_no_female',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r101-sam1-s46'
    # )

    # create_embeddings_for_all(
    #     path='../results/chexpert_no_old/kd-feature_rgb/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert_no_old',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r101-sam1-s46'
    # )

    # create_embeddings_for_all(
    #     path='../results/chexpert_no_white/kd-feature_rgb/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert_no_white',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r101-sam1-s46'
    # )

    # # 2. HAM10000

    # print('Feature-KD Models HAM10000')

    # create_embeddings_for_all(
    #     path='../results/ham10000_no_female/kd-feature_rgb/',
    #     sensitive_attributes=['Age', 'Sex'],
    #     data_name='ham10000_no_female',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r101-sam1-s42'
    # )

    # create_embeddings_for_all(
    #     path='../results/ham10000_no_old/kd-feature_rgb/',
    #     sensitive_attributes=['Age', 'Sex'],
    #     data_name='ham10000_no_old',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r101-sam1-s42'
    # )


    # #### ATTENTION-KD MODELS HIGH CAPACITY

    # # 1. CHEXPERT

    # print('Attention-KD Models CheXpert')

    # create_embeddings_for_all(
    #     path='../results/chexpert_no_female/attention_kd/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert_no_female',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r101-sam1-s46'
    # )

    # create_embeddings_for_all(
    #     path='../results/chexpert_no_old/attention_kd/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert_no_old',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r101-sam1-s46'
    # )

    # create_embeddings_for_all(
    #     path='../results/chexpert_no_white/attention_kd/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert_no_white',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r101-sam1-s46'
    # )

    # # 2. HAM10000

    # print('Attention-KD Models HAM10000')

    # create_embeddings_for_all(
    #     path='../results/ham10000_no_female/attention_kd/',
    #     sensitive_attributes=['Age', 'Sex'],
    #     data_name='ham10000_no_female',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r101-sam1-s42'
    # )

    # create_embeddings_for_all(
    #     path='../results/ham10000_no_old/attention_kd/',
    #     sensitive_attributes=['Age', 'Sex'],
    #     data_name='ham10000_no_old',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r101-sam1-s42'
    # )


    # ------------- OLD -------------

    # Original Models
    # create_embeddings_for_all(
    #     path='../results/chexpert/resnet/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert',
    #     num_workers=workers,
    #     gpu=0,
    # )

    # No Female
    # create_embeddings_for_all(
    #     path='../results/chexpert_no_female/resnet/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert_no_female',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True
    # )

    # create_embeddings_for_all(
    #     path='../results/chexpert_no_female/kd/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert_no_female',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r34-sam1-s43'
    # )

    # create_embeddings_for_all(
    #     path='../results/chexpert_no_female/kd-feature_rgb/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert_no_female',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r34-sam1-s43'
    # )

    # # No Old
    # create_embeddings_for_all(
    #     path='../results/chexpert_no_old/resnet/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert_no_old',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True
    # )

    # create_embeddings_for_all(
    #     path='../results/chexpert_no_old/kd/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert_no_old',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r34-sam1-s43'
    # )

    # create_embeddings_for_all(
    #     path='../results/chexpert_no_old/kd-feature_rgb/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert_no_old',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r34-sam1-s43'
    # )

    # No White
    # create_embeddings_for_all(
    #     path='../results/chexpert_no_white/resnet/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert_no_white',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True
    # )

    # create_embeddings_for_all(
    #     path='../results/chexpert_no_white/kd/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert_no_white',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r34-sam1-s43'
    # )

    # create_embeddings_for_all(
    #     path='../results/chexpert_no_white/kd-feature_rgb/',
    #     sensitive_attributes=['Age', 'Sex', 'Race'],
    #     data_name='chexpert_no_white',
    #     num_workers=workers,
    #     gpu=0,
    #     is_student=True,
    #     is_kd=True,
    #     teacher_id='r34-sam1-s43'
    # )