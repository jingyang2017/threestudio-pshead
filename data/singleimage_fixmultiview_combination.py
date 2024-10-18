import math
import os
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.typing import *
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .uncond_multiview import (
    FixMultiviewCameraDataModuleConfig,
    FixMultiviewCameraIterableDataset,
    MultiviewCameraDataset
)
from threestudio.data.image import (
    SingleImageDataModuleConfig,
    SingleImageIterableDataset,
)


class SingleImageMultiViewCameraIterableDataset(IterableDataset, Updateable):
    def __init__(
        self, cfg_single_view: Any, cfg_multi_view: Any,cfg_four_view: Any, cfg_front_view:Any, prob_multi_view: int = None) -> None:
        super().__init__()
        self.cfg_single = parse_structured(
            SingleImageDataModuleConfig, cfg_single_view
        )
        self.cfg_multi = parse_structured(
            FixMultiviewCameraDataModuleConfig, cfg_multi_view
        )

        self.cfg_4view = parse_structured(
            FixMultiviewCameraDataModuleConfig, cfg_four_view
        )

        self.cfg_front = parse_structured(
            FixMultiviewCameraDataModuleConfig, cfg_front_view)

        self.train_dataset_single = SingleImageIterableDataset(self.cfg_single,'test')
        self.train_dataset_multi = FixMultiviewCameraIterableDataset(self.cfg_multi)
        # print(self.cfg_multi.azimuth_range)
        # self.cfg_multi.azimuth_range = [0,360]
        # self.train_dataset_multi_2 = FixMultiviewCameraIterableDataset(self.cfg_multi)
        self.train_dataset_4view = FixMultiviewCameraIterableDataset(self.cfg_4view)
        self.train_dataset_front = FixMultiviewCameraIterableDataset(self.cfg_front)

        self.idx = 0
        self.prob_multi_view = prob_multi_view

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # self.train_dataset_single.update_step(epoch, global_step, on_load_weights)
        self.train_dataset_multi.update_step(epoch, global_step, on_load_weights)
        self.train_dataset_4view.update_step(epoch, global_step, on_load_weights)
        self.train_dataset_front.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}
    

    def collate(self, batch) -> Dict[str, Any]:
        
        if self.prob_multi_view is not None:
            multi = random.random() < self.prob_multi_view
        else:
            multi = False
        
        if multi:
            # batch =  self.train_dataset_4view.collate(batch)
            batch =  self.train_dataset_multi.collate(batch)
            if random.random()>0.5:
                batch["is_video"] = True
            else:                
                # batch =  self.train_dataset_front.collate(batch)
                batch["is_video"] = False

            # batch["is_video"] = False
            # batch["single_view"] = False
            batch["single_view"] = False

            # rand = random.random()
            # if rand < 0.3:
            #     batch =  self.train_dataset_multi.collate(batch)
            # elif rand < 0.6:
            #     # if self.train_dataset_multi.height==512:
            #         # batch =  self.train_dataset_front.collate(batch)
            #     # else:
            # batch['is_video'] = True
            # else:
            #     batch =  self.train_dataset_4view.collate(batch)
            
            # batch =  self.train_dataset_multi.collate(batch)
            # if random.random()>0.5:
            #     batch["is_video"] = True  #vid diffusion
            # else:
            #     batch["is_video"] = False #zero123
        else:
            batch = self.train_dataset_single.collate(batch)
            batch["single_view"] = True
        self.idx += 1
        return batch


@register("singleimage-multiview-combined-camera-datamodule")
class SingleImageFixMultiviewCombinedCameraDataModule(pl.LightningDataModule):
    cfg: FixMultiviewCameraDataModuleConfig
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        # print(cfg)
        self.cfg = cfg
        self.cfg_single = parse_structured(SingleImageDataModuleConfig, cfg.single_view)
        self.cfg_multi = parse_structured(FixMultiviewCameraDataModuleConfig, cfg.multi_view)
        self.cfg_4view = parse_structured(FixMultiviewCameraDataModuleConfig, cfg.four_view)
        self.cfg_front = parse_structured(FixMultiviewCameraDataModuleConfig, cfg.front_view)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = SingleImageMultiViewCameraIterableDataset(
                self.cfg_single,
                self.cfg_multi,
                self.cfg_4view,
                self.cfg_front,
                prob_multi_view=self.cfg.prob_multi_view,
            )
        if stage in [None, "fit", "validate"]:
            self.val_dataset = MultiviewCameraDataset(self.cfg_multi)
        if stage in [None, "test", "predict"]:
            self.test_dataset = MultiviewCameraDataset(self.cfg_multi)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=None, collate_fn=self.val_dataset.collate
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=None, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=None, collate_fn=self.test_dataset.collate
        )

    # def val_dataloader(self) -> DataLoader:
    #     return self.general_loader(
    #         self.val_dataset, batch_size=self.val_dataset.n_view, collate_fn=self.val_dataset.collate
    #     )

    # def test_dataloader(self) -> DataLoader:
    #     return self.general_loader(
    #         self.test_dataset, batch_size=self.test_dataset.n_view, collate_fn=self.test_dataset.collate
    #     )

    # def predict_dataloader(self) -> DataLoader:
    #     return self.general_loader(
    #         self.test_dataset, batch_size=self.test_dataset.n_view, collate_fn=self.test_dataset.collate
    #     )
