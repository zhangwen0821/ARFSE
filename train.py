import os
import argparse
from argparse import ArgumentParser
from datetime import datetime
import pytz
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from flowmse.backbones.shared import BackboneRegistry
from flowmse.data_module import SpecsDataModule
from flowmse.model import VFModel


kst = pytz.timezone('Asia/Shanghai') 
formatted_time = datetime.now(kst).strftime("%Y%m%d%H%M%S") 

def get_argparse_groups(parser, args): 
    groups = {}
    for group in parser._action_groups:
        groups[group.title] = argparse.Namespace(**{
            a.dest: getattr(args, a.dest, None) for a in group._group_actions
        })
    return groups

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")    
    temp_args, _ = parser.parse_known_args()
    backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
    data_module_cls = SpecsDataModule

    pl.Trainer.add_argparse_args(parser)
    VFModel.add_argparse_args(parser.add_argument_group("VFModel"))
    backbone_cls.add_argparse_args(parser.add_argument_group("Backbone"))
    data_module_cls.add_argparse_args(parser.add_argument_group("DataModule"))
    
    args = parser.parse_args()
    arg_groups = get_argparse_groups(parser, args)

    model = VFModel(
        backbone=args.backbone, 
        data_module_cls=data_module_cls,
        **{
            **vars(arg_groups['VFModel']),
            **vars(arg_groups['Backbone']),
            **vars(arg_groups['DataModule'])
        }
    )


    dataset = os.path.basename(os.path.normpath(args.base_dir))
    logger = TensorBoardLogger(
        save_dir="logs", 
        name=f"dataset_{dataset}", 
        version=f"{model.method}_{formatted_time}" 
    )
    model_dir = os.path.join(logger.save_dir, logger.name, logger.version, "checkpoints")

    callbacks = [
        ModelCheckpoint(dirpath=model_dir, save_last=True, filename='{epoch}_last'),
        ModelCheckpoint(dirpath=model_dir, save_top_k=5, monitor="pesq", mode="max", filename='{epoch}_{pesq:.2f}'),
        ModelCheckpoint(dirpath=model_dir, save_top_k=5, monitor="si_sdr", mode="max", filename='{epoch}_{si_sdr:.2f}')
    ]

    trainer = pl.Trainer.from_argparse_args(
        arg_groups['pl.Trainer'],
        accelerator='gpu', 
        strategy="ddp",
        logger=logger, 
        log_every_n_steps=10,
        num_sanity_val_steps=1,
        max_epochs=100,
        callbacks=callbacks
    )

    trainer.fit(model)