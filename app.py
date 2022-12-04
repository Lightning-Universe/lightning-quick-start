import os.path as ops
import lightning as L
from quick_start.components import PyTorchLightningScript, ImageServeGradio

class TrainDeploy(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.train_work = PyTorchLightningScript(
            script_path=ops.join(ops.dirname(__file__), "./train_script.py"),
            script_args=["--trainer.max_epochs=10"],
            cloud_compute=L.CloudCompute("cpu-medium", idle_timeout=60),
        )

        self.serve_work = ImageServeGradio(start_with_flow=False)

    def run(self):
        # 1. Run the python script that trains the model
        self.train_work.run()

        # 2. when a checkpoint is available, deploy
        if self.train_work.best_model_path:
            self.serve_work.run(self.train_work.best_model_path)

    def configure_layout(self):
        tabs = []
        if not self.train_work.has_stopped:
            tabs.append({"name": "Model training", "content": self.train_work})
        tabs.append({"name": "Interactive demo", "content": self.serve_work})
        return tabs

app = L.LightningApp(TrainDeploy())
