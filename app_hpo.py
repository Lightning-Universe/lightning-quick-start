import os.path as ops
import lightning as L
from quick_start.components import PyTorchLightningScript, ImageServeGradio
import optuna
from lightning_hpo import Optimizer, BaseObjective

class HPOPyTorchLightningScript(PyTorchLightningScript, BaseObjective):

    @staticmethod
    def distributions():
        return {"model.lr": optuna.distributions.LogUniformDistribution(0.0001, 0.1)}


class TrainDeploy(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.train_work = Optimizer(
            script_path=ops.join(ops.dirname(__file__), "./train_script.py"),
            script_args=["--trainer.max_epochs=5"],
            objective_cls=HPOPyTorchLightningScript,
            n_trials=4
        )

        self.serve_work = ImageServeGradio(L.CloudCompute("cpu"))

    def run(self):
        # 1. Run the python script that trains the model
        self.train_work.run()

        # 2. when a checkpoint is available, deploy
        if self.train_work.best_model_path:
            self.serve_work.run(self.train_work.best_model_path)

    def configure_layout(self):
        tab_1 = {"name": "Model training", "content": self.train_work.hi_plot}
        tab_2 = {"name": "Interactive demo", "content": self.serve_work}
        return [tab_1, tab_2]

app = L.LightningApp(TrainDeploy())
