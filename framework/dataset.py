from enum import Enum
from typing import Literal, Mapping, Tuple

import numpy as np
import pandas as pd
import torch.utils.data as data_utils
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from typing_extensions import TypeGuard
from collections import defaultdict
from collections import Counter
import timeit
from PIL import Image
import torchvision.transforms as transforms


class SamplingTechnique(Enum):
    """Sampling techniques used and their ids"""
    NO_SAMPLING = 0
    RANDOM_WEIGHTED_SAMPLING = 1
    SUBGROUP_SAMPLING = 2


class SensitiveAttribute(Enum):
    """All datasets use one or more of these attributes and are binarized in the
    following way. We expect the csv files to have a column with the same name as
    the Key field in the mapping below."""

    sex = {0: "Male", 1: "Female", "Key": "Sex"}
    age = {0: "0-60", 1: "60+", "Key": "Age"}
    race = {0: "White", 1: "Non-White", "Key": "Race"}
    skintype = {0: "Types 1-3", 1: "Types 4-6", "Key": "SkinType"}


class AvailableDataset(Enum):
    """Register datasets by adding them, their directories 
    and available sensitive attributes here."""
    chexpert = {
        'sensitive_attributes': [SensitiveAttribute.sex, 
                                 SensitiveAttribute.age, 
                                 SensitiveAttribute.race],
        'split_dir': 'data/chexpert/splits',
        'data_dir': '../../../../../biodata/data/chest_xray/CheXpert-v1.0/preproc_224x224'
    }
    ham10000 = {
        'sensitive_attributes': [SensitiveAttribute.sex, 
                                 SensitiveAttribute.age],
        'split_dir': 'data/ham10000/splits',
        'data_dir': 'data/ham10000/images',
    }
    fitzpatrick17k = {
        'sensitive_attributes': [SensitiveAttribute.skintype],
        'split_dir': 'data/fitzpatrick17k/splits',
        'data_dir': 'data/fitzpatrick17k/preproc_224x224',
    }

    ##### Unfair datasets #####
    
    ## Ham10000
    ham10000_no_male = {
        'sensitive_attributes': [SensitiveAttribute.sex, 
                                 SensitiveAttribute.age],
        'split_dir': 'data/ham10000_no_male/',
        'data_dir': 'data/ham10000/images',
    }
    ham10000_no_female = {
        'sensitive_attributes': [SensitiveAttribute.sex, 
                                 SensitiveAttribute.age],
        'split_dir': 'data/ham10000_no_female/',
        'data_dir': 'data/ham10000/images',
    }
    ham10000_no_old = {
        'sensitive_attributes': [SensitiveAttribute.sex,
                                SensitiveAttribute.age],
        'split_dir': 'data/ham10000_no_old/',
        'data_dir': 'data/ham10000/images',
    }
    ham10000_no_young = {
        'sensitive_attributes': [SensitiveAttribute.sex,
                                SensitiveAttribute.age],
        'split_dir': 'data/ham10000_no_young/',
        'data_dir': 'data/ham10000/images',
    }
    ham10000_equal = {
        'sensitive_attributes': [SensitiveAttribute.sex, 
                                 SensitiveAttribute.age],
        'split_dir': 'data/ham10000_equal/',
        'data_dir': 'data/ham10000/images',
    }
    ham10000_upsample = {
        'sensitive_attributes': [SensitiveAttribute.sex, 
                                 SensitiveAttribute.age],
        'split_dir': 'data/ham10000_upsample/',
        'data_dir': 'data/ham10000/images',
    }
    ham10000_equal_upsample = {
        'sensitive_attributes': [SensitiveAttribute.sex, 
                                 SensitiveAttribute.age],
        'split_dir': 'data/ham10000_equal_upsample/',
        'data_dir': 'data/ham10000/images',
    }
    ham10000_equal_upsample_2 = {
        'sensitive_attributes': [SensitiveAttribute.sex, 
                                 SensitiveAttribute.age],
        'split_dir': 'data/ham10000_equal_upsample_2/',
        'data_dir': 'data/ham10000/images',
    }

    ## CheXpert
    chexpert_no_male = {
        'sensitive_attributes': [SensitiveAttribute.sex, 
                                 SensitiveAttribute.age, 
                                 SensitiveAttribute.race],
        'split_dir': 'data/chexpert_no_male',
        'data_dir': '../../../../../biodata/data/chest_xray/CheXpert-v1.0/preproc_224x224'
    }
    chexpert_no_female = {
        'sensitive_attributes': [SensitiveAttribute.sex, 
                                 SensitiveAttribute.age, 
                                 SensitiveAttribute.race],
        'split_dir': 'data/chexpert_no_female',
        'data_dir': '../../../../../biodata/data/chest_xray/CheXpert-v1.0/preproc_224x224'
    }
    chexpert_no_old = {
        'sensitive_attributes': [SensitiveAttribute.sex, 
                                 SensitiveAttribute.age, 
                                 SensitiveAttribute.race],
        'split_dir': 'data/chexpert_no_old',
        'data_dir': '../../../../../biodata/data/chest_xray/CheXpert-v1.0/preproc_224x224'
    }
    chexpert_no_young = {
        'sensitive_attributes': [SensitiveAttribute.sex, 
                                 SensitiveAttribute.age, 
                                 SensitiveAttribute.race],
        'split_dir': 'data/chexpert_no_young',
        'data_dir': '../../../../../biodata/data/chest_xray/CheXpert-v1.0/preproc_224x224'
    }
    chexpert_no_white = {
        'sensitive_attributes': [SensitiveAttribute.sex, 
                                 SensitiveAttribute.age, 
                                 SensitiveAttribute.race],
        'split_dir': 'data/chexpert_no_white',
        'data_dir': '../../../../../biodata/data/chest_xray/CheXpert-v1.0/preproc_224x224'
    }
    chexpert_no_non_white = {
        'sensitive_attributes': [SensitiveAttribute.sex, 
                                 SensitiveAttribute.age, 
                                 SensitiveAttribute.race],
        'split_dir': 'data/chexpert_no_non_white',
        'data_dir': '../../../../../biodata/data/chest_xray/CheXpert-v1.0/preproc_224x224'
    }
    chexpert_equal = {
        'sensitive_attributes': [SensitiveAttribute.sex, 
                                 SensitiveAttribute.age, 
                                 SensitiveAttribute.race],
        'split_dir': 'data/chexpert_equal',
        'data_dir': '../../../../../biodata/data/chest_xray/CheXpert-v1.0/preproc_224x224'
    }
    chexpert_upsample = {
        'sensitive_attributes': [SensitiveAttribute.sex, 
                                 SensitiveAttribute.age, 
                                 SensitiveAttribute.race],
        'split_dir': 'data/chexpert_upsample',
        'data_dir': '../../../../../biodata/data/chest_xray/CheXpert-v1.0/preproc_224x224'
    }


def is_valid_sensitive_attribute(
    a: SensitiveAttribute, ds: AvailableDataset
) -> TypeGuard[SensitiveAttribute]:
    if a not in ds.value['sensitive_attributes']:
        raise ValueError(f"Invalid sensitive attribute {a} for dataset {ds}")
    return True

def is_valid_dataset(ds: AvailableDataset) -> TypeGuard[AvailableDataset]:
    if ds not in AvailableDataset.__members__.keys():
        raise ValueError(f"Invalid dataset {ds}")
    return True


class FairnessDataset(Dataset):
    def __init__(
        self,
        split_dir: str,
        data_dir: str,
        split: Literal["train", "val", "test"],
        sensitive_attributes: list[SensitiveAttribute],
        dataset_name: AvailableDataset,
    ) -> None:
        super().__init__()
        for attribute in sensitive_attributes:
            assert is_valid_sensitive_attribute(attribute, dataset_name)

        split_path = split_dir + f"/{split}.csv"
        self.split_df = pd.read_csv(split_path)
        
        # drop bad images
        # self.split_df = self.split_df[self.split_df["image_id"] != "ISIC_0029145"]
        # self.split_df = self.split_df[self.split_df["image_id"] != "ISIC_0029424"]

        self.data_dir = data_dir
        self.sensitive_attributes = sensitive_attributes

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self.data_dir + "/" + self.split_df.iloc[idx]["image_id"] + ".jpg"

        # try:
        #     with Image.open(img_path) as img:
        #         img = transforms.ToTensor()(img)
        # except Exception as e:
        #     print(f"Error in reading image at {img_path}")
        #     raise e

        try:
            img = torchvision.io.read_image(
                img_path, mode=torchvision.io.image.ImageReadMode.RGB
            )
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            raise e

        label = self.split_df.iloc[idx]["binaryLabel"]
        label = torch.tensor(label, dtype=torch.long)

        subgroups = {}
        for attribute in self.sensitive_attributes:
            subgroup_key = attribute.value["Key"]
            subgroup_value = torch.tensor(self.split_df.iloc[idx][subgroup_key], dtype=torch.long)
            subgroups[subgroup_key] = subgroup_value

        return img, label, subgroups
    
    def __len__(self):
        return len(self.split_df)


class FairnessDataModule(pl.LightningDataModule):
    def __init__(
            self,
            sensitive_attributes: list[str],
            dataset_name: str,
            batch_size: int,
            num_workers: int,
            sampling_technique: SamplingTechnique = SamplingTechnique.NO_SAMPLING,
    ):
        
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampling_technique = sampling_technique

        assert is_valid_dataset(dataset_name)
        self.dataset_name = AvailableDataset[dataset_name.lower()]
        self.split_dir = self.dataset_name.value['split_dir']
        self.data_dir = self.dataset_name.value['data_dir']
        self.sensitive_attributes = [SensitiveAttribute[attribute.lower()] for attribute in sensitive_attributes]


    def setup(self, stage: str) -> None:
        # Asserts for valid inputs
        assert stage in ("fit", "test", "validate", "predict")
        for attribute in self.sensitive_attributes:
            assert is_valid_sensitive_attribute(attribute, self.dataset_name)

        if stage == "fit":
            self.train_dataset = FairnessDataset(
                self.split_dir,
                self.data_dir,
                "train",
                self.sensitive_attributes,
                self.dataset_name,
            )
            self.val_dataset = FairnessDataset(
                self.split_dir,
                self.data_dir,
                "val",
                self.sensitive_attributes,
                self.dataset_name,
            )
        elif stage == "validate":
            self.val_dataset = FairnessDataset(
                self.split_dir,
                self.data_dir,
                "val",
                self.sensitive_attributes,
                self.dataset_name,
            )
        elif stage == "predict":
            self.val_dataset = FairnessDataset(
                self.split_dir,
                self.data_dir,
                "val",
                self.sensitive_attributes,
                self.dataset_name,
            )
            self.test_dataset = FairnessDataset(
                self.split_dir,
                self.data_dir,
                "test",
                self.sensitive_attributes,
                self.dataset_name,
            )
        elif stage == "test":
                self.test_dataset = FairnessDataset(
                self.split_dir,
                self.data_dir,
                "test",
                self.sensitive_attributes,
                self.dataset_name,
            )

    ### ANOTHER WAY TO CALCULATE IT
    def calculate_weights_based_on_labels(self):
        labels = self.train_dataset.split_df['binaryLabel'].to_numpy()
        label_counts = np.bincount(labels)
        label_weights = 1.0 / label_counts[labels]
        return label_weights
    
    def calculate_weights_based_on_subgroups(self):
        subgroup_weights = []
        for attribute in self.sensitive_attributes:
            attribute_key = attribute.value["Key"]
            labels = np.array([x[2][attribute_key] for x in self.train_dataset])
            subgroup_counts = np.bincount(labels)
            subgroup_class_weights = 1.0 / subgroup_counts
            subgroup_weights.append(subgroup_class_weights[labels])
        return subgroup_weights

    def train_dataloader(self) -> DataLoader:
        if self.sampling_technique == SamplingTechnique.NO_SAMPLING:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
            )
        else:
            if self.sampling_technique == SamplingTechnique.RANDOM_WEIGHTED_SAMPLING:
                weights = self.calculate_weights_based_on_labels()
            elif self.sampling_technique == SamplingTechnique.SUBGROUP_SAMPLING:
                weights = self.calculate_weights_based_on_labels()
                subgroup_weights = self.calculate_weights_based_on_subgroups()
                for subgroup_weight in subgroup_weights:
                        weights *= subgroup_weight

            sampler = data_utils.WeightedRandomSampler(
                weights=weights,
                num_samples=len(self.train_dataset),
                replacement=True
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sampler=sampler
            )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
    
    def predict_dataloader(self) -> DataLoader:    
        test_loader = DataLoader(
                    self.test_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=False,
                  )
        val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=False,
                  )
        return [test_loader, val_loader]
    
    def get_subgroup_names(self) -> Mapping[int, str]:
        return self.sensitive_attribute.value
