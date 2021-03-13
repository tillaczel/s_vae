## GPU env

### Prerequisites
[Docker](https://www.docker.com/) <br/>
[Nvidia drivers](https://www.nvidia.com/Download/index.aspx)

### Build and run Docker
In the terminal navigate to project folder and type:
```
docker build -t s_vae .
docker run --rm -it --name='training' -v PATH_TO_PROHJECT/s_vae:/s_vae --gpus all s_vae
```
Note you need to replace PATH_TO_PROHJECT with your path.

## Run training
```
cd s_vae/experiment/train
python run.py 
```

## Start TensorBoard
All the experiment results can be seen in TensorBoard. To start TensorBoard type:
```
docker run --rm -it --name='tensorboard' -p:8888:5678 -v PATH_TO_PROHJECT/s_vae:/s_vae s_vae
tensorboard --logdir=/s_vae/local/logs/name/ --host 0.0.0.0 --port 5678
```
Go to browser and type:
```
localhost:8888
```