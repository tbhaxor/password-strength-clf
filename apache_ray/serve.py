import os

os.environ["KERAS_BACKEND"] = "jax"

from ray import serve
from argparse import RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter
from pathlib import Path
from utils import CLASS_NAMES, DEFAULT_CHARSET, get_model, get_vectorized
from keras import ops
from starlette.requests import Request


class MixedFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


BASE_DIR = Path(__file__).parent.parent.relative_to(Path.cwd())
MODEL_PATH = BASE_DIR / "trained-models" / "best_model"


@serve.deployment(
    num_replicas=int(os.getenv("NUM_REPLICAS", "1")),
    ray_actor_options={
        "num_gpus": int(os.getenv("NUM_GPUS", "1")),
    },
)
class PasswordClassifier:
    def __init__(self):
        self.model = get_model((len(DEFAULT_CHARSET),), model_path=MODEL_PATH)

    def classify(self, password: str) -> str:
        password_vectorized = get_vectorized(DEFAULT_CHARSET, [password])
        class_id = ops.argmax(self.model(password_vectorized))
        return CLASS_NAMES[class_id]

    async def __call__(self, request: Request):
        payload = await request.json()
        password = payload.get("password")
        if not password or type(password) is not str:
            return {"statusCode": 400, "message": "passwords are required"}

        return {"statusCode": 200, "strength": self.classify(password)}

    pass


clf = PasswordClassifier.bind()
