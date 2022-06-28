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

### Learn how it works

The components are [here](https://github.com/Lightning-AI/lightning-quick-start/blob/main/quick_start/components.py) and the code is heavily commented. 

Once you understand well this example, you aren't a beginner with Lightning App anymore 🔥
