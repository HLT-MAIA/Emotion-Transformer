# Emotion-Transformer
Emotion Detection with Transformer models

## Install:

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