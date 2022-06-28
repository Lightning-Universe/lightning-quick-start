import warnings
warnings.simplefilter("ignore")
import logging
import os
from functools import partial
import torch
import torchvision.transforms as T
from quick_start.download import download_data
from subprocess import Popen
from lightning.app.storage import Path
from lightning.app.components.python import TracerPythonScript
from lightning.app.components.serve import ServeGradio
import gradio as gr

logger = logging.getLogger(__name__)


class PyTorchLightningScript(TracerPythonScript):

    """This component executes a PyTorch Lightning script
    and injects a callback in the Trainer at runtime in order to start tensorboard server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 1. Keep track of the best model path.
        self.best_model_path = None
        self.best_model_score = None

    def configure_tracer(self):
        # 1. Override `configure_tracer``

        # 2. Import objects from pytorch_lightning
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import Callback

        # 3. Create a tracer.
        tracer = super().configure_tracer()

        # 4. Implement a callback to launch tensorboard server.
        class TensorboardServerLauncher(Callback):

            def __init__(self, work):
                # The provided `work` is the current ``PyTorchLightningScript`` work.
                self._work = work

            def on_train_start(self, trainer, *_):
                # Provide `host` and `port` in order for tensorboard to be usable in the cloud.
                self._work._process = Popen(
                    f"tensorboard --logdir='{trainer.logger.log_dir}' --host {self._work.host} --port {self._work.port}",
                    shell=True,
                )

        def trainer_pre_fn(self, *args, work=None, **kwargs):
            # Intercept Trainer __init__ call and inject a ``TensorboardServerLauncher`` component.
            kwargs['callbacks'].append(TensorboardServerLauncher(work))
            return {}, args, kwargs

        # 5. Patch the `__init__` method of the Trainer to inject our callback with a reference to the work.
        tracer.add_traced(Trainer, "__init__", pre_fn=partial(trainer_pre_fn, work=self))
        return tracer

    def run(self, *args, **kwargs):
        ######### [DEMO PURPOSE] #########

        # 1. Download a pre-trained model for speed reason.
        download_data("https://pl-flash-data.s3.amazonaws.com/assets_lightning/demo_weights.pt", "./")

        # 2. Add some arguments to the Trainer to make training faster.
        self.script_args += [
            "--trainer.limit_train_batches=12",
            "--trainer.limit_val_batches=4",
            "--trainer.callbacks=ModelCheckpoint",
            "--trainer.callbacks.monitor=val_acc",
        ]

        # 3. Utilities
        warnings.simplefilter("ignore")
        logger.info(f"Running train_script: {self.script_path}")
        ######### [DEMO PURPOSE] #########

        logger.info(f"Running train_script: {self.script_path}")

        # 4. Execute the parent run method
        super().run(*args, **kwargs)

    def on_after_run(self, script_globals):
        # 1. Once the script has finished to execute, we can collect its globals and access any objects.
        # Here, we are accessing the LightningCLI and the associated lightning_module
        lightning_module = script_globals["cli"].trainer.lightning_module

        # 2. From the checkpoint_callback, we are accessing the best model weights
        checkpoint = torch.load(script_globals["cli"].trainer.checkpoint_callback.best_model_path)

        # 3. Load the best weights and torchscript the model.
        lightning_module.load_state_dict(checkpoint["state_dict"])
        lightning_module.to_torchscript("model_weight.pt")

        # 4. Use lightning.app.storage.Path to create a reference to the torchscripted model
        # When running in the cloud on multiple machines, by simply passing this reference to another work,
        # it triggers automatically a transfer.
        self.best_model_path = Path("model_weight.pt")

        # 5. Keep track of the metrics.
        self.best_model_score = float(script_globals["cli"].trainer.checkpoint_callback.best_model_score)

class ImageServeGradio(ServeGradio):

    inputs = gr.inputs.Image(type="pil", shape=(28, 28))
    outputs = gr.outputs.Label(num_top_classes=10)

    def __init__(self, cloud_compute, *args, **kwargs):
        super().__init__(*args, cloud_compute=cloud_compute, **kwargs)
        self.examples = None
        self.best_model_path = None
        self._transform = None
        self._labels = {idx: str(idx) for idx in range(10)}

    def run(self, best_model_path):
        ######### [DEMO PURPOSE] #########
        # Download some examples so it works locally and in the cloud (issue with gradio on loading the images.)
        download_data("https://pl-flash-data.s3.amazonaws.com/assets_lightning/images.tar.gz", "./")
        self.examples = [os.path.join(str("./images"), f) for f in os.listdir("./images")]
        ######### [DEMO PURPOSE] #########

        self.best_model_path = best_model_path
        self._transform = T.Compose([T.Resize((28, 28)), T.ToTensor()])
        super().run()

    def predict(self, img):
        # 1. Receive an image and transform it into a tensor
        img = self._transform(img)[0]
        img = img.unsqueeze(0).unsqueeze(0)

        # 2. Apply the model on the image and convert the logits into probabilities
        prediction = torch.exp(self.model(img))

        # 3. Return the data in the `gr.outputs.Label` format
        return {self._labels[i]: prediction[0][i].item() for i in range(10)}

    def build_model(self):
        # 1. Load the best model. As torchscripted by the first component, using torch.load works out of the box.
        model = torch.load(self.best_model_path)

        # 2. Prepare the model for predictions.
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        # 3. Return the model.
        return model