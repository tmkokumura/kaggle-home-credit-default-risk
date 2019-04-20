# How to use
## Git checkout
```
git checkout git@github.com:tmkokumura/kaggle-home-credit-default-risk.git
```

## Install depending modules
```
pip install -f requirements.txt
```

## Preprocess


## Train
### Simple training of LightGBM model.
```
python train_lgbm.py
```

### Cross validation training of LightGBM model.
```
python train_lgbm_cv.py
```


### Grid search of LightGBM model.
```
python train_lgbm_gs.py
```

###  Simple training of DNN model.
```
python train_nn.py
```