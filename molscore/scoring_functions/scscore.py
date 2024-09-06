import logging

from molscore import resources
from molscore.scoring_functions.base import BaseServerSF

logger = logging.getLogger("scscore")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class SCScore(BaseServerSF):
    """
    Predicted synthetic complexity according to https://doi.org/10.1021/acs.jcim.7b00622
    """

    return_metrics = ["score"]
    model_dictionary = {
        "1024bool": resources.files("molscore.data.models.SCScore").joinpath(
            "full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.pickle"
        ),
        "1024unit8": resources.files("molscore.data.models.SCScore").joinpath(
            "full_reaxys_model_1024uint8/model.ckpt-10654.as_numpy.pickle"
        ),
        "2048bool": resources.files("molscore.data.models.SCScore").joinpath(
            "full_reaxys_model_2048bool/model.ckpt-10654.as_numpy.pickle"
        ),
    }

    def __init__(
        self,
        prefix: str = "SCScore",
        env_engine: str = "mamba",
        model: str = "1024bool",
        server_grace: int = 60,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance
        :param env_engine: Environment engine [conda, mamba]
        """
        assert (
            model in self.model_dictionary
        ), f"Model not found in {self.model_dictionary}"
        model_path = self.model_dictionary[model]
        assert model_path.exists(), f"Model file not found at {model_path}"

        super().__init__(
            prefix=prefix,
            env_engine=env_engine,
            env_name="scscore-env",
            env_path=resources.files("molscore.data.models.SCScore").joinpath(
                "environment.yml"
            ),
            server_path=resources.files("molscore.scoring_functions.servers").joinpath(
                "scscore_server.py"
            ),
            server_grace=server_grace,
            server_kwargs={"model_path": model_path},
        )
