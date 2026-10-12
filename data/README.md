# Dataset setup

Use the Mendeley Data chest X-ray archive described in the root README: [dataset landing page](https://data.mendeley.com/datasets/rscbjbr9sj/2), citation DOI 10.17632/rscbjbr9sj.2, license shown as CC BY 4.0 on the dataset record. Verify current terms and cite Kermany, Zhang, and Goldbaum. The code never downloads the 1.15 GB archive automatically.

After reviewing the dataset terms, you may explicitly download the listed ChestXRay2017 archive in PowerShell from the repository root (this command transfers about 1.15 GB):

```powershell
New-Item -ItemType Directory -Force .\data\download | Out-Null
Invoke-WebRequest -Uri "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded" -OutFile .\data\download\ChestXRay2017.zip
Expand-Archive -LiteralPath .\data\download\ChestXRay2017.zip -DestinationPath .\data\extracted
Get-ChildItem .\data\extracted -Directory -Recurse
```

Inspect the extracted folders and set `--dataset` to the directory containing `train`, `val`, and `test`; the archive may include a nested directory. Do not flatten or merge the supplied splits blindly.

Extract so `data/chest_xray/` contains:

```text
train/NORMAL/       train/PNEUMONIA/
val/NORMAL/         val/PNEUMONIA/
test/NORMAL/        test/PNEUMONIA/
```

The image folders are local and ignored by Git. Do not commit the data archive or image files. Preserve the given test split; missing validation data is an error, not replaced by test data. Use `--dataset PATH` to point to another compatible root. See `src/data_loader.py` for validation behavior.
