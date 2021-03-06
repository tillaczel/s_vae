## GPU env

### Prerequisites
[Docker](https://www.docker.com/) <br/>
[Nvidia drivers](https://www.nvidia.com/Download/index.aspx)

### Build Docker

GPU:
```
cd docker/GPU
docker build -t s_vae .
```

CPU:
```
cd docker/CPU
docker build -t s_vae_cpu .
```

## Run training
GPU:
```
docker run --rm -it --name='training' -v PATH_TO_PROJECT:/s_vae --gpus all s_vae
cd s_vae/experiment/train
python train.py 
```
Note you need to replace PATH_TO_PROJECT with your path.<br/>

CPU:
Set `gpu: 0` in the [config file](experiment/config.yaml).
```
docker run --rm -it --name='training' -v PATH_TO_PROHJECT:/s_vae s_vae_cpu
conda deactivate
cd s_vae/experiment/train
python train.py 
```
Note you need to replace PATH_TO_PROJECT with your path.<br/>


## Start TensorBoard
All the experiment results can be seen in TensorBoard. To start TensorBoard type:
```
docker run --rm -it --name='tensorboard' -p:8888:5678 -v PATH_TO_PROHJECT:/s_vae s_vae
tensorboard --logdir=/s_vae/local/logs/name/ --host 0.0.0.0 --port 5678
```
Go to browser and type:
```
localhost:8888
```
