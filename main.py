import os
from argparse import ArgumentParser

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import wandb
import models
from utils import SaveNodeEncodings


def main(args):
    logname = '{}_{}_{}_seed-{}'.format(args.dataset, args.model_name, args.base_model_name, args.seed)
    if args.suffix != '':
        logname += '-{}'.format(args.suffix)
    wandb_logger = WandbLogger(name=logname, project='fedgnn', save_dir='artifacts')
    seed_everything(args.seed)
    early_stop_callback = EarlyStopping(
        monitor='val/loss', patience=args.early_stop_patience, mode='min', verbose=True
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss', save_top_k=1, save_last=True, mode='min', verbose=True
    )
    trainer = Trainer.from_argparse_args(args,
        default_root_dir='artifacts',
        deterministic=True, 
        logger=wandb_logger,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        gpus=args.gpus,
        callbacks=[SaveNodeEncodings()]
    )
    
    if args.train:
        model = getattr(models, args.model_name)(args)
        if args.restore_train_ckpt_path != '':
            trainer = Trainer(
                resume_from_checkpoint=args.restore_train_ckpt_path,
                default_root_dir='artifacts',
                deterministic=True, 
                logger=wandb_logger,
                early_stop_callback=early_stop_callback,
                checkpoint_callback=checkpoint_callback,
                gpus=args.gpus,
                callbacks=[SaveNodeEncodings()]
            )
        trainer.fit(model)
    
    if args.load_test_ckpt_path != '':
        load_test_ckpt_path = args.load_test_ckpt_path
    else:
        load_test_ckpt_path = checkpoint_callback.best_model_path
    print(load_test_ckpt_path)
    model = getattr(models, args.model_name).load_from_checkpoint(
        load_test_ckpt_path,
        save_node_encodings_test=args.save_node_encodings_test
    )
    trainer.test(model)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='NodeClassifier', help='name of the model')
    parser.add_argument('--base_model_name', type=str, default='MLP')
    parser.add_argument('--prop_model_name', type=str, default='')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--save_node_encodings_test', action='store_true')
    parser.add_argument('--load_test_ckpt_path', type=str, default='')
    parser.add_argument('--notrain', dest='train', action='store_false')
    parser.add_argument('--restore_train_ckpt_path', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=50)
    parser = Trainer.add_argparse_args(parser)
    temp_args, _ = parser.parse_known_args()
    ModelClass = getattr(models, temp_args.model_name)
    BaseModelClass = getattr(models, temp_args.base_model_name, None)
    PropModelClass = getattr(models, temp_args.prop_model_name, None)
    parser = ModelClass.add_model_specific_args(parser)
    if BaseModelClass is not None:
        parser = BaseModelClass.add_model_specific_args(parser)
    if PropModelClass is not None:
        parser = PropModelClass.add_model_specific_args(parser)

    args = parser.parse_args()
    main(args)