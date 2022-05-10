import lightning as L

class TrainDeploy(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.train_work = L.components.demo.PyTorchLightningScript(
            script_path="./train_script.py",
            script_args=["--trainer.max_epochs=5"],
        )

        self.serve_work = L.components.demo.ImageServeGradio(L.CloudCompute("cpu", 1))

    def run(self):
        # 1. Run the python script that trains the model
        self.train_work.run()

        # 2. when a checkpoint is available, deploy
        if self.train_work.best_model_path:
            self.serve_work.run(self.train_work.best_model_path)

    def configure_layout(self):
        tab_1 = {"name": "WandB Run", "content": self.train_work}
        tab_2 = {"name": "Gradio Demo", "content": self.serve_work}
        return [tab_1, tab_2]

app = L.LightningApp(TrainDeploy())