import os.path as ops
import lightning as L
from quick_start.components import PyTorchLightningScript, ImageServeGradio
from lightning_hpo import OptunaPythonScript
import optuna



class DistributionsPyTorchLightningScript(PyTorchLightningScript):

    @staticmethod
    def distributions():
        return {"model.lr": optuna.distributions.LogUniformDistribution(0.0001, 0.1)}


class TrainDeploy(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.train_work = OptunaPythonScript(
            script_path= ops.join(ops.dirname(__file__), "./train_script.py"),
            script_args=["--trainer.max_epochs=5"],
            total_trials=4,
            objective_work_cls=DistributionsPyTorchLightningScript,
        )

        self.serve_work = ImageServeGradio(L.CloudCompute("cpu"))

    def run(self):
        # 1. Run the python script that trains the model
        self.train_work.run()

        # 2. when a checkpoint is available, deploy
        if self.train_work.best_model_path:
            self.serve_work.run(self.train_work.best_model_path)

    def configure_layout(self):
        tabs = self.train_work.configure_layout()
        serve_tab = {"name": "Interactive demo", "content": self.serve_work}
        return tabs + [serve_tab]

app = L.LightningApp(TrainDeploy())
