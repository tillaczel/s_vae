## GPU env

### Prerequisites
[Docker](https://www.docker.com/) <br/>
[Nvidia drivers](https://www.nvidia.com/Download/index.aspx)

### Build and run Docker
```
docker build -t s_vae .
docker run --rm -it -v PATH_TO_PROHJECT/s_vae:/s_vae --gpus all s_vae
```
Note you need to replace PATH_TO_PROHJECT with your path.

## Run experiment
```
cd s_vae/experiment
python run.py 
```