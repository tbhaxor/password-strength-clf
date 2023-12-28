import os

os.environ["KERAS_BACKEND"] = "jax"

from argparse import ArgumentParser
from pathlib import Path
from keras import ops
from utils import (
    DEFAULT_CHARSET,
    MixedFormatter,
    TrainArgs,
    fit_model,
    get_model,
    get_one_hot,
    get_vectorized,
    read_data,
    split_dataset,
    CLASS_NAMES,
)

BASE_DIR = Path(__file__).parent.relative_to(Path.cwd())

parser = ArgumentParser(description="Password strength classifier trainer script", formatter_class=MixedFormatter)
parser.add_argument(
    "file",
    type=Path,
    help="path of the csv dataset file",
)
parser.add_argument(
    "--model-path",
    type=Path,
    metavar="",
    default=BASE_DIR / "trained-models" / "best_model",
    help="path to save only weights of the best model",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    metavar="",
    default=0.01,
    help="learning rate for adam optimizer",
)
parser.add_argument(
    "--dropout-rate",
    default=0.3,
    type=float,
    metavar="",
    help="dropout rate to be used between last hidden layer and the output layer",
)
parser.add_argument(
    "--epochs",
    type=int,
    metavar="",
    default=5,
    help="number of epochs to train for",
)
parser.add_argument(
    "--batch-size",
    type=int,
    metavar="",
    default=32,
    help="batch size to use",
)
parser.add_argument(
    "--vocab",
    type=str,
    metavar="CHARACTERS",
    help="vocabulary for the passwords to train the model on",
),
parser.add_argument(
    "--validation-split",
    type=float,
    default=0.2,
    metavar="",
    help="validation split during model training",
)
parser.add_argument(
    "--passwords",
    nargs="+",
    type=str,
    metavar="PASSWORD",
    help="sample passwords to predict and show the performance",
)

args = TrainArgs(**vars(parser.parse_args()))
CHARSET = args.vocab or DEFAULT_CHARSET

if not args.file.exists():
    raise ValueError(f"Dataset file {args.file} does not exist!")

if args.model_path.suffix != ".weights.h5":
    args.model_path = args.model_path.with_suffix(".weights.h5")

df = read_data(args.file)
print("Sample of each class")
print(df.groupby("strength").sample(1))

X = get_vectorized(CHARSET, df["password"])
y = get_one_hot(df["strength"])
print("Features shape:", X.shape)
print("Labels shape:", y.shape)

X_train, X_test, y_train, y_test = split_dataset(X, y)
print("Number samples for training:", X_train.shape[0])
print("Number samples for testing:", X_test.shape[0])

model = get_model(X_train.shape[1:], args.learning_rate, args.dropout_rate)
model.summary()
best_model = fit_model(
    X,
    y,
    args.validation_split,
    args.batch_size,
    args.epochs,
    args.model_path,
    model,
)

print("Evaluating the best model")
evaluation = best_model.evaluate(X_test, y_test, return_dict=True)
print(f"\tLoss: {evaluation['loss']}")
print(f"\tAccuracy: {evaluation['categorical_accuracy']}")
print(f"\tPrecision: {evaluation['precision']}")
print(f"\tRecall: {evaluation['recall']}")

if args.passwords:
    print("Running user provided tests")
    X_sample_vectorized = get_vectorized(CHARSET, args.passwords)
    y_sample_predicted = best_model.predict(X_sample_vectorized, verbose=False)
    y_class_ids = ops.argmax(y_sample_predicted, axis=1)

    for password, class_id in zip(args.passwords, y_class_ids):
        print(f"Password: {password:30}\tStrength: {CLASS_NAMES[class_id]}")
