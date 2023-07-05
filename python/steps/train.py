__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

import pytorch_lightning as pl
import torch

from data.rachel.dataloader import DataLoader
from project import Project


def main(proj: Project):
    # Reproducibility
    proj.reproducible()

    # Instantiate Dataloader
    dataloaders = proj.prepare_dataloader()

    # Model
    model = proj.prepare_model()

    # Logger
    list_loggers = proj.create_loggers()

    # Callbacks
    list_callbacks = proj.create_callbacks()

    # Trainer
    trainer = pl.Trainer(max_epochs=proj.max_epochs,
                         num_nodes=proj.num_gpus,
                         accelerator=proj.accelerator,
                         logger=list_loggers,
                         callbacks=list_callbacks,
                         check_val_every_n_epoch=1,
                         num_sanity_val_steps=0,
                         gradient_clip_val=proj.grad_clip_val)

    # Train
    trainer.fit(model=model,
                train_dataloaders=dataloaders.train_loader,
                val_dataloaders=dataloaders.dev_loader)

    # Test
    trainer.test(model=model, dataloaders=dataloaders.test_loader)
