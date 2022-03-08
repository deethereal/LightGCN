# PyTorch-LightGCN

LightGCN for everyone! This is an adaptaion of the [current](https://github.com/gusye1234/LightGCN-PyTorch) PyTorch implimentation.  [Paper](https://arxiv.org/pdf/2002.02126.pdf)
## Enviroment Requirement

`pip install -r requirements.txt`
## Dataset

You can load your own dataset, using `LoadOwnData` class from `dataloader.py` or using `to_valid_format.py` (the second method is faster). After preprocessing your data register it with `register.py`.

## Config
Set the config of your model in the `code/config.yml`

## Change base directory

Change `ROOT_PATH` in `code/world.py`

## Run
Now just type the following command
`cd code && python main.py`

*NOTE from the original repo*:

1. Even though we offer the code to split user-item matrix for matrix multiplication, we strongly suggest you don't enable it since it will extremely slow down the training speed.
2. If you feel the test process is slow, try to increase the `test_u_batch_size` and enable `multicore`(Windows system may encounter problems with `multicore` option enabled)
