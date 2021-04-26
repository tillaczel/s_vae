import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from s_vae.engine import EngineModule
from s_vae.data import create_dataset


def crecte_ckpt_callback(ckpt_config: dict):
    monitor = ckpt_config['monitor']
    save_top_k = ckpt_config['save_top_k']
    return ModelCheckpoint(
        filename='{epoch}',
        save_top_k=save_top_k,
        monitor=monitor,
        mode='min',
    )


def create_data_loaders(config: dict):
    train_set, valid_set, test_set = create_dataset(config['data'], config['experiment']['seed'])

    batch_size = config['training'].get('batch_size', 32)
    num_workers = config['data'].get('num_workers', 1)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader


def create_trainer(config: dict):
    tb_logger = loggers.TestTubeLogger(config['experiment']['save_dir'], name=config['experiment']['name'])
    _callbacks = [crecte_ckpt_callback(config['training']['ckpt_callback'])]
    trainer = pl.Trainer(logger=tb_logger,
                         gpus=config['gpu'],
                         max_epochs=config['training']['max_epochs'],
                         progress_bar_refresh_rate=20,
                         deterministic=True,
                         terminate_on_nan=True,
                         num_sanity_val_steps=0,
                         callbacks=_callbacks
                         )
    return trainer


def train(config: dict):
    pl.seed_everything(config['experiment'].get('seed', 8756))

    train_loader, valid_loader, test_loader = create_data_loaders(config)

    engine = EngineModule(config)
    trainer_fix_var = create_trainer(config)
    trainer_fix_var.fit(model=engine, train_dataloader=train_loader, val_dataloaders=valid_loader)
    trainer_fix_var.test(test_dataloaders=valid_loader)

    engine.set_fix_var(False)
    trainer = create_trainer(config)
    trainer.fit(model=engine, train_dataloader=train_loader, val_dataloaders=valid_loader)
    trainer.test(test_dataloaders=valid_loader)

