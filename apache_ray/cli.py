import os

os.environ["KERAS_BACKEND"] = "jax"

from argparse import ArgumentParser, RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter
import ray
from pathlib import Path
from keras import ops
from utils import DEFAULT_CHARSET, CLASS_NAMES, get_model, get_vectorized


class MixedFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


BASE_DIR = Path(__file__).parent.parent.relative_to(Path.cwd())

parser = ArgumentParser(
    description="Distributed inference of the password classifier using Apache Ray cluster",
    formatter_class=MixedFormatter,
)
parser.add_argument(
    "--model-path",
    type=Path,
    metavar="",
    default=BASE_DIR / "trained-models" / "best_model",
    help="path to save only weights of the best model",
)
parser.add_argument(
    "--vocab",
    type=str,
    metavar="CHARACTERS",
    help="vocabulary for the passwords to train the model on",
)
parser.add_argument(
    "-c",
    type=int,
    default=None,
    help="Number of CPU cores to allocate",
    metavar="NUM_CPUs",
    dest="num_cpus",
)
parser.add_argument(
    "-g",
    type=int,
    default=None,
    help="Number of GPUs to allocate",
    metavar="NUM_GPUs",
    dest="num_gpus",
)
parser.add_argument(
    "passwords",
    nargs="+",
    help="Provide the passwords to run inference on and return the classification results",
    metavar="PASSWORD [...PASSWORD]",
)

args = parser.parse_args()
CHARSET = args.vocab or DEFAULT_CHARSET

ray.init()


@ray.remote(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
class Actor:
    def __init__(self, passwords: list[str]):
        self.passwords = passwords
        self.vectorized_inputs = get_vectorized(CHARSET, passwords)
        self.model = get_model(self.vectorized_inputs.shape[1:], model_path=args.model_path)

    def predict(self):
        y_predictions = self.model.predict(self.vectorized_inputs, verbose=False)
        return list(zip(self.passwords, ops.argmax(y_predictions, axis=1)))


actor = Actor.remote(args.passwords)

for password, klass in ray.get(actor.predict.remote()):
    print(f"{password}: {CLASS_NAMES[klass]}")
