import logging

from lightning import LightningApp, LightningFlow

from quick_start import PyTorchLightningScript, ImageServeGradio

logger = logging.getLogger(__name__)


class RootFlow(LightningFlow):
    def __init__(self):
        super().__init__()

    def run(self):

        print(PyTorchLightningScript)
        print(ImageServeGradio)
        exit(0)


app = LightningApp(RootFlow())
