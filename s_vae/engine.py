import pytorch_lightning as pl
import torch

from s_vae.models import build_model


class EngineModule(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model = build_model(config['model'])

    @property
    def lr(self):
        return self.optimizers().param_groups[0]['lr']

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss_metrics = self.model.step(x)
        self.log('lr', self.lr, prog_bar=True, on_step=True, logger=False)
        return loss_metrics

    def training_epoch_end(self, outputs: list):
        self.transform_and_log_results(outputs, 'train')

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss_metrics = self.model.step(x)
        return loss_metrics

    def validation_epoch_end(self, outputs: list):
        self.transform_and_log_results(outputs, 'valid')

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss_metrics = self.model.step(x)
        return loss_metrics

    def test_epoch_end(self, outputs: list):
        self.transform_and_log_results(outputs, 'test')

    def configure_optimizers(self):
        optimizer = get_optimizer(self.config['training']['optimizer'], self.parameters())
        scheduler_config = self.config['training'].get('scheduler', None)
        if scheduler_config is not None:
            scheduler = get_scheduler(scheduler_config, optimizer)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def transform_and_log_results(self, outputs, split):
        metrics = dict()
        for key in outputs[0].keys():
            metrics[f'{split}/{key}'] = torch.stack([x[key] for x in outputs]).mean()
        self.logger.log_metrics(metrics, step=self.current_epoch)

        _callback_metrics = [self.config['training']['scheduler']['monitor'],
                             self.config['training']['ckpt_callback']['monitor']]
        for _callback_metric in _callback_metrics:
            if _callback_metric in metrics.keys():
                self.trainer.logger_connector.callback_metrics[_callback_metric] = metrics[_callback_metric]


def get_optimizer(optim_config: dict, params):
    name = optim_config['name']
    lr = optim_config['lr']

    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr)
    elif name == 'adam':
        return torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError(f'{name} not in optimizers')


def get_scheduler(scheduler_config, optimizer):
    name = scheduler_config['name']
    monitor = scheduler_config['monitor']

    if name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode=scheduler_config['mode'],
                                                               patience=scheduler_config['patience'],
                                                               factor=scheduler_config['factor'],
                                                               min_lr=scheduler_config['min_lr'])
        return dict(scheduler=scheduler, monitor=monitor)
    else:
        raise ValueError(f'{name} not in schedulers')

