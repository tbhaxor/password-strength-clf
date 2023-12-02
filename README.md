# Password Strength Classifier

An example of Jax usage with the new Keras 3.0 API release

## Why this Project?

Being a fan of [keras framework](https://keras.io/about/), the anouncement of [Keras 3](https://keras.io/keras_3/) is a big YEAH! moment for me. I can now experiment with other deep learning frameworks, like [Pytorch](https://pytorch.org/) or the [Jax](https://jax.readthedocs.io/), with just two or three lines of modification in the current codebase. Yeah, it is no longer limited to using TensorFlow alone!

Any Keras model that only uses [built-in layers](https://keras.io/2.15/api/layers/) will immediately work with all supported backends. In fact, your existing tf.keras models that only use [built-in layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers) can start running in JAX and PyTorch right away! That's right, your codebase just gained a whole new set of capabilities.

<p align=center>

<img width=400 height=350 src="https://media.tenor.com/8zHzYq3eBVwAAAAd/baby-scream-yeah.gif">

</p>

I've been using Jax because it's the only one that works on CUDA 12 local setup without the need for extra cudnn downloads. I intend to create a simple password strength classifier project to show how simple it is to configure the jax backend for the keras.


## TL;DR

1. Install the CUDA 12.x compatible jax and latest version of keras from pip
    ```console
    pip install -U pip
    pip install -U "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install keras
    ```
2. Configure the backend before keras import
    ```python
    import os
    os.environ["KERAS_BACKEND"] = "jax"

    import keras
    ```

    > **Note** If you used environment variables outside the code, you can use `keras.backend.backend()` to retrive the name of the backend.

## What's new

Although, Keras 3 is a total rewrite of the previous codebase. These are a few of the highlights I want to share with you.

1. **Framework agnostic** &mdash; You can pick the framework that suits you best, and switch from one to another based on your current goals. 
2. **Model parallism** &mdash; It was [previously accomplished with Tensorflow](https://www.tensorflow.org/guide/distributed_training), but now it includes [distribution namespace](https://keras.io/guides/distribution/), making it simple to perform model parallelism, data parallelism, and combinations of the two. 
3. **Universal data pipelines** &mdash; The Keras 3 `fit()` / `evaluate()` / `predict()` routines are compatible with [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) objects, with PyTorch [`DataLoader`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) objects, with [NumPy arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html), [Pandas dataframes](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) &dash; regardless of the backend you're using.
4. **Ops namespace** &mdash; In case you don't want to extend the existing layers, you can use [`keras.ops`](https://keras.io/api/ops/) to create components (like arbitrary custom layers or pretrained models) that works across all backends. It provides numpy API (Not something "_NumPy-like_" &dash; just literally the NumPy AP) and neural-network functions (softmax, conv and what not).

[**And more...**](https://keras.io/keras_3/)

## Requirements

- Python 3.11.x
- (Optional) Poetry 1.7.0
- CUDA 12.2, cuDNN 8.9, NCCL 2.16

## Setup

1. Clone the repository

    ```console
    git clone --depth=1 https://github.com/tbhaxor/password-strength-clf.git
    cd password-strength-clf
    ```

2. (Optional: Using pip) Configure and activate the virtual environment

    ```console
    virtualenv venv
    source venv/bin/activate
    ```

3. Install the dependencies

    ```console
    pip install -r requirements.txt
    ```
    Or, with poetry
    ```console
    poetry install
    ```

    > **Note** Poetry automatically creates a virtual environment if it does not exists, and installs the packages into it.

4. Download the [dataset](https://www.kaggle.com/datasets/bhavikbb/password-strength-classifier-dataset/data) from the kaggle or create the CSV in format with 2 columns
    
    |Column Name|Order|Description|
    |:--:|:---:|:---:|
    |`password`|1|Textual password in raw format, hashed password are not allowed.|
    |`strength`|2|Strength class of the corresponding password where **0** means WEAK, **1** means MODERATE and **2** means STRONG.|

## Training the Model

The [`train.py`](#training-customization) script takes the input CSV file as required  

```console
$ python train.py /path/to/data.csv --passwords HELLO hello h3ll000 H3ll00W@rld h3LL00@@102030
....
Running user provided tests
Password: HELLO                                 Strength: WEAK
Password: hello                                 Strength: WEAK
Password: h3ll000                               Strength: WEAK
Password: H3ll00W@rld                           Strength: MODERATE
Password: h3LL00@@102030                        Strength: STRONG
```

> **Note** As a bare minimum, only the path to the input CSV file is required to begin training and save the best model for it. However, I recommend that you use the `--passwords` argument to predict the classes using the best model for the custom passwords.

## Training Customization

Use "`python train.py -h`" to get the arguments provided by train.py script

```console
usage: train.py [-h] [--model-path] [--learning-rate] [--dropout-rate]
                [--epochs] [--batch-size] [--vocab CHARACTERS]
                [--validation-split] [--passwords PASSWORD [PASSWORD ...]]
                file

Password strength classifier trainer script

positional arguments:
  file                  path of the csv dataset file

options:
  -h, --help            show this help message and exit
  --model-path          path to save only weights of the best model (default: trained-models/best_model)
  --learning-rate       learning rate for adam optimizer (default: 0.01)
  --dropout-rate        dropout rate to be used between last hidden layer and the output layer (default: 0.3)
  --epochs              number of epochs to train for (default: 5)
  --batch-size          batch size to use (default: 32)
  --vocab CHARACTERS    vocabulary for the passwords to train the model on (default: None)
  --validation-split    validation split during model training (default: 0.2)
  --passwords PASSWORD [PASSWORD ...]
                        sample passwords to predict and show the performance (default: None)
```

## Using Pretrained Model

If you've already [trained](#training-the-model) the model and want to use it in your application, use the following code.

```py
from utils import read_data, get_vectorized, get_model, CLASS_NAMES
from keras import ops
from pathlib import Path

X = get_vectorized(CHARSET, ["your password"])
saved_model = Path("/path/to/weights")
if not saved_model.exists():
    raise FileNotFoundError(f"Path '{saved_model}' does not exist!")

model = get_model(X.shape[1:], model_path="/path/to/weights")

y_predicted = model.predict(X, verbose=False)
y_class_id = ops.argmax(y_predicted, axis=1)
y_class = CLASS_NAMES[y_class_id[0]]
```

## Contact Me

Email: tbhaxor _at_ gmail _dot_ com <br />
Discord: @tbhaxor.com <br />
Twitter: @tbhaxor <br />
LinkedIn: @tbhaxor

