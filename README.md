## GPU env

### Prerequisites
[Docker](https://www.docker.com/) <br/>
[Nvidia drivers](https://www.nvidia.com/Download/index.aspx)

### Build Docker
In the terminal navigate to project folder and type:
```
docker build -t s_vae .
```

## Run training
```
docker run --rm -it --name='training' -v PATH_TO_PROHJECT:/s_vae --gpus all s_vae
cd s_vae/experiment/train
python run.py 
```
Note you need to replace PATH_TO_PROHJECT with your path.<br/>
If you don't have a GPU remove `--gpus all` from the `docker run` command and set `gpu: 0` in the [config file](experiment/config.yaml).

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
