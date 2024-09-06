import argparse
import logging
import os

from flask import Flask, jsonify, request
from scscore.standalone_model_numpy import SCScorer
from utils import get_mol

logger = logging.getLogger("scscore_server")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

app = Flask(__name__)


class Model:
    """
    This particular class uses the QSAR model from SCScore in a stand alone environment to avoid conflict dependencies.
    """

    def __init__(self, model_path: os.PathLike, **kwargs):
        """
        :param model_path: Path to pre-trained model (specifically clf.pkl)
        """
        self.scorer = SCScorer()
        self.scorer.restore(model_path)  # todo: support other models


@app.route("/", methods=["POST"])
def compute():
    # Get smiles from request
    smiles = request.get_json().get("smiles", [])

    # Handle invalid as this is not handled by MolSkill
    valids = []
    for i, smi in enumerate(smiles):
        if get_mol(smi):
            valids.append(i)

    # Make predictions
    scores = model.scorer.get_scores([smiles[i] for i in valids])

    # Update results
    results = [{"smiles": smi, "score": 0.0} for smi in smiles]
    for i, score in zip(valids, scores):
        results[i].update({"score": float(score)})

    # Return results
    return jsonify(results)


def get_args():
    parser = argparse.ArgumentParser(description="Run a scoring function server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to pre-trained model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model = Model(**args.__dict__)
    app.debug = False
    app.run(port=args.port)
