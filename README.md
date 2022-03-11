## Targeted Sentiment Analysis for Norwegian

This repository provides the data, baseline code, and extras necessary to begin work on the targeted sentiment track for IN5550. Cloning this repo is meant as a quick way of getting something working, but there are many ways of improving these results, ranging from small technical changes (including a hyperparameter search, more/less regularization, small architecture modifications) to larger and more theoretical changes (Comparing model architectures, adding character-level information, or using transfer learning models). Feel free to change anything necessary in the code.

## Usage

```
python baseline.py --NUM_LAYERS number of hidden layers for BiLSTM \\
                   --HIDDEN_DIM dimensionality of LSTM layers \\
                   --BATCH_SIZE number of examples to include in a batch \\
                   --DROPOUT dropout to be applied after embedding layer \\
                   --EMBEDDING_DIM dimensionality of embeddings \\
                   --EMBEDDINGS location of pretrained embeddings \\
                   --TRAIN_EMBEDDINGS whether to train or leave fixed \\
                   --LEARNING_RATE learning rate for ADAM optimizer \\
                   --EPOCHS number of epochs to train model
```

Note that with the current code, you have to provide the model with pretrained embeddings. All other parameters can be left as default values.

## Requirements

1. Python 3
2. sklearn  ```pip install -U scikit-learn```
3. Pytorch ```pip install torch torchvision torchtext```
4. tqdm ```pip install tqdm```
5. torchtext ```pip install torchtext```

