# Descriotion
This repository includes codes for Kaggle Home Credit Default Risk competition.
This model reached 0.79565 of area under the ROC curve, and won bronz medal.

# How to use
## Git clone
```
git clone git@github.com:tmkokumura/kaggle-home-credit-default-risk.git
```

## Install depending modules
```
pip install -f requirements.txt
```

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
