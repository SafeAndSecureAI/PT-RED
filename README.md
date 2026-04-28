# NC / PT-RED

Brief instructions to run `NC.py` and `pt-red.py` on CIFAR-10 models.

## Requirements
- Python 3.8+
- See `requirements.txt`

## Install
```bash
pip install -r requirements.txt
```

## Available model dirs
These come from the `*_model.pth` files in this folder:
- `badnet`
- `blend`
- `1pixel`
- `chess`
- `clean`

## Ground truth target class
- `badnet: 9`
- `blend: 8`
- `1pixel: 4`
- `chess: 5`
- `clean: none`

## Run
From this folder:

```bash
python NC.py --model_dir badnet
python pt-red.py --model_dir badnet
```

Example with another model:
```bash
python NC.py --model_dir clean
```

Notes:
- The scripts download CIFAR-10 to `./data` on first run.
- `pt-red.py` creates `./<model_dir>/pert_estimated` for outputs.
