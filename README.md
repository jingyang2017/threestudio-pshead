# threestudio-pshead

## install env
```
conda env create -f pshead_env.yml
```

## install threestudio

```
https://github.com/threestudio-project/threestudio
```

### clone following repos

```
cd custom
https://github.com/DSaurus/threestudio-3dgs
https://github.com/DSaurus/threestudio-4dfy
```

# preprocess images
```
preprocess_image.py
```
## dreambooth and blip
```
bash train_dreambooth.sh 
```
## download face detection, coderfomer and recognition model.
See pretrained/download

## threestudio
```
sh custom/threestudio-3dface/train.sh 
sh custom/threestudio-3dface/test.sh 
```