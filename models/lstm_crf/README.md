# bi-LSTM + CRF

__Architecture__

1. [GloVe 840B vectors](https://nlp.stanford.edu/projects/glove/)
2. Bi-LSTM
3. CRF

__Related Paper__ [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991) by Huang, Xu and Yu

__Training time__ ~ 20 min

|| `train` | `testa` | `testb` | Paper, `testb` |
|---|:---:|:---:|:---:|:---:|
|best | 98.45 |93.81 | __90.61__ |  90.10 |
|best (EMA)| 98.82 | 94.06 | 90.43 | |
|mean ± std| 98.85 ± 0.22| 93.68 ± 0.12| 90.42 ± 0.10|  |
|mean ± std (EMA)| 98.71 ± 0.47 | 93.81 ± 0.24 | __90.50__ ± 0.21| |
|abs. best |   | | 90.61 |  |
|abs. best (EMA) | |  | 90.75 |  |


## 1. Download glove

![Data Format](../../images/data.png)


```
cd data/example
make download-glove
```


## 2. Run `python main.py`

Using python3, it will run training with early stopping and write predictions under `results/score/{name}.preds.txt`.

## 3. Run the `conlleval` script on the predictions

Usage: 
```
    cd models/lstm_crf/
    ../conlleval < ./results/score/test.preds.txt > ./results/score/test.metrics.txt
```

## 4. Run `python interact.py`

It reloads the trained estimator and computes predicitons on a simple example

## 5. Run `python export.py`

It exports the estimator inference graph under `saved_model`. We will re-use this for serving.

## 6. Run `python serve.py`

It reloads the saved model using `tf.contrib.predictor` and computes prediction on a simple example.
