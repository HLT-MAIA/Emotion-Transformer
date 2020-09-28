# Emotion-Transformer
Emotion Detection with Transformer models 😃😡😱😊

To test our model capacity to predict emotions we use the [GoEmotions Corpus](https://www.aclweb.org/anthology/2020.acl-main.372.pdf). This corpus consists of 58k reddit comments annotated with 28 different emotions.

# Model Architecture

<div style="text-align:center"><img src="resources/transformer.png" alt="architecture"></div>

Our model is built on top of a pretrained Transformer model such as RoBERTa. To get a sentence representation we apply a pooling technique (average, max or CLS) and pass that representation to a classification head that produces an independent score for each label.


# Install

```bash
virtualenv -p python3.6 emot-env
source emot-env/bin/activate

https://github.com/HLT-MAIA/Emotion-Transformer
cd Emotion-Transformer
pip install -r requirements.txt
```

## Command Line Interface:

### Train:

To set up your training you have to define your model configs. Take a look at the `example.yaml` in the configs folder, where all hyperparameters are briefly described.

After defining your hyperparameter run the following command:
```bash
python cli.py train -f configs/example.yaml
```

### Monitor training with Tensorboard:
Launch tensorboard with:

```
tensorboard --logdir="experiments/"
```


## Interact:
Fun command where we can interact with with a trained model.

```bash
python cli.py interact --experiment experiments/{experiment_id}/
```

## Testing:
After training we can test the model against the testset by running.

```bash
python cli.py test --experiment experiments/{experiment_id}/
```

This will compute the precision, recall and F1 for each label and the macro-average results.

## Results:

### Ekman Emotion

| Model | Macro-Precision | Macro-Recall | Macro-F1 |
| :---: | :---: | :---: | :---: |
| biLSTM [Reported](https://arxiv.org/pdf/2005.00547.pdf) | - | - | 0.53 | 
| BERT-base [Reported](https://arxiv.org/pdf/2005.00547.pdf) | 0.59 | 0.69 | 0.64 |
| Mini-BERT | 0.43 | 0.69 | 0.51 |
| RoBERTa-base | 0.58 | 0.69 | 0.62 |

**Note:** The results reported were achieved with default parameters. With some search over hyper-parameters better results can be achieved.
