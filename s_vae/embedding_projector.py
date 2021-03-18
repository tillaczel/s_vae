import torch
from torch.utils.data import DataLoader

from s_vae.data import create_dataset
from s_vae.models import build_model


def run(config: dict):
    ep_config = config['embedding_projector']
    log_dir = f"{config['experiment']['save_dir']}/{config['experiment']['name']}/version_{ep_config['version']}"
    checkpoint_path = f"{log_dir}/checkpoints/epoch={ep_config['epoch']}.ckpt"
    '/s_vae/local/logs/name/version_6/checkpoints/epoch=9.ckpt'
    model = build_model(config['model']).eval()
    checkpoint = torch.load(checkpoint_path)['state_dict']
    _checkpoint = dict()
    for key in checkpoint.keys():
        new_key = key.replace('model.', '')
        _checkpoint[new_key] = checkpoint[key]
    model.load_state_dict(_checkpoint)

    train_set, valid_set, test_set = create_dataset(config['data'], config['experiment']['seed'])

    batch_size = config['training'].get('batch_size', 32)
    num_workers = config['data'].get('num_workers', 1)

    valid_set = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    x_data, y_data, embeddings = list(), list(), list()
    for x, y in valid_set:
        x_hat, z = model.forward(x)
        x_data.append(x)
        y_data.append(y)
        embeddings.append(z)
    x_data, y_data, embeddings = torch.cat(x_data), torch.cat(y_data).detach().numpy(), torch.cat(embeddings)

    if len(x.shape) < 4:
        label_img = None
    else:
        label_img = x_data

    writer = torch.utils.tensorboard.writer.SummaryWriter(log_dir=f'{log_dir}/embedding_projector')
    writer.add_embedding(embeddings, metadata=y_data, label_img=label_img)

