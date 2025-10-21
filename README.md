# CEHAFNet

Enhanced Saliency Detection in Remote Sensing: A Hierarchical Adaptive Fusion Approach



## Environment


| CPU        | GPU  | OS         | CUDA Version |
| ---------- | ---- | ---------- | ------------ |
| i7-12700KF | 3090 | Ubuntu2204 | 12.4         |


```
python==3.10
pytorch==2.4.0
torchvision==0.19.0
cudatoolkit==11.3
softadapt==0.0.5
noise==1.2.2
tensorboard==1.15.0
tqdm
thop
pyyaml
bytecode==0.16.0
```



## Datasets

| Dataset Name | Source URL                                                                                                          |
|-------------|---------------------------------------------------------------------------------------------------------------------|
| ORSSD   | [Github](https://github.com/rmcong/ORSSD-dataset) or [BaiduNetdisk](https://pan.baidu.com/s/1k44UlTLCW17AS0VhPyP7JA) |
| EORSSD   | [DAFNet](https://github.com/rmcong/EORSSD-dataset)                                                                  |


## Train

After modifying the TODO tags in Train_AdpMLL.py, run it directly.

## Test

After modifying the `image_dir` and `prediction_dir` in mtest.py, run it directly.

## Contact

If you have any questions, please contact us by email lisa21230402@henu.edu.cn.
