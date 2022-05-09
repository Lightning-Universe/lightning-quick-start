# Lightning Quick Start App

### Install Lightning

```bash
pip install lightning --extra-index-url https://tinyurl.com/n2scv9p4
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
