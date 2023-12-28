import string
from argparse import (
    ArgumentDefaultsHelpFormatter,
    RawDescriptionHelpFormatter,
    RawTextHelpFormatter,
)
from dataclasses import dataclass
from pathlib import Path
from typing import AnyStr, Optional, Tuple

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Input, ReLU
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy, Precision, Recall
from keras.models import Model
from keras.optimizers import Adam
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


class MixedFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


DEFAULT_CHARSET = string.ascii_letters + string.digits + string.punctuation + string.whitespace[:1]


@dataclass
class TrainArgs:
    file: Path
    model_path: Path
    epochs: int
    batch_size: int
    vocab: Optional[AnyStr]
    validation_split: int
    learning_rate: float
    passwords: Tuple[AnyStr]
    dropout_rate: Optional[float]


CLASS_NAMES = ["WEAK", "MODERATE", "STRONG"]


def get_vectorized(charset: str, inputs) -> np.ndarray:
    vectorizer = CountVectorizer(vocabulary=list(charset), analyzer="char_wb")
    return vectorizer.transform(inputs).toarray()


def get_one_hot(series: pd.Series) -> np.ndarray:
    return pd.get_dummies(series, drop_first=False).to_numpy()


def read_data(filepath: Path, drop_na=True, equal_distribution=True) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        on_bad_lines=lambda args: [",".join(args[:-1]), int(args[-1])],
        engine="python",
    )

    if drop_na:
        df = df.dropna()

    if equal_distribution:
        strength_count = df["strength"].value_counts()
        df = df.groupby("strength").sample(strength_count.min())

    return df


def split_dataset(
    X: pd.Series,
    y: pd.Series,
    shuffle=True,
    random_state=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if shuffle and random_state is None:
        random_state = np.random.randint(1000, 9999)

    return train_test_split(
        X,
        y,
        train_size=0.70,
        shuffle=shuffle,
        random_state=random_state,
    )


def get_model(
    input_shape: Tuple[int],
    learning_rate: Optional[float] = None,
    dropout_rate: Optional[float] = None,
    model_path: Optional[Path] = None,
) -> Model:
    input_ = Input(shape=input_shape)
    h = Dense(128)(input_)
    h = ReLU()(h)
    h = Dense(256)(h)
    h = ReLU()(h)
    h = Dense(512)(h)
    h = ReLU()(h)
    if dropout_rate is not None:
        h = Dropout(rate=dropout_rate)(h)
    output = Dense(3)(h)

    model = Model(inputs=input_, outputs=output, name="password_strength")
    if model_path and model_path.exists():
        model.load_weights(model_path)

    if learning_rate is None:
        optimizer = "adam"
    else:
        optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(from_logits=True),
        metrics=[
            Precision(name="precision"),
            Recall(name="recall"),
            CategoricalAccuracy(name="categorical_accuracy"),
        ],
    )

    return model


def fit_model(
    X: np.ndarray,
    y: np.ndarray,
    validation_split: float,
    batch_size: int,
    epochs: int,
    save_path: str,
    model: Model,
) -> Model:
    model.fit(
        X,
        y,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            EarlyStopping(patience=3),
            ModelCheckpoint(save_path, save_best_only=True, save_weights_only=True),
        ],
    )

    return get_model(
        X.shape[1:],
        model_path=save_path,
    )
