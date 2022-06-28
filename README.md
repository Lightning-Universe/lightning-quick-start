# Lightning Quick Start App

### Install Lightning

```bash
pip install lightning
```

### Locally

In order to run the application locally, run the following commands

```
pip install -r requirements.txt
lightning run app app.py
```

### Cloud

In order to run the application cloud, run the following commands

### On CPU

```
lightning run app app.py --cloud
```

### On GPU

```
USE_GPU=1 lightning run app app.py --cloud
```

### Adding HPO support to Quick Start App.

Using [Lightning HPO](https://github.com/Lightning-AI/LAI-lightning-hpo-App), you can easily convert the training component into a Sweep Component.

```bash
pip install lightning-hpo
lightning run app app_hpo.py
```

### Learn how it works

The components are [here](https://github.com/Lightning-AI/lightning-quick-start/blob/main/quick_start/components.py) and the code is heavily commented.

Once you understand well this example, you aren't a beginner with Lightning App anymore 🔥
