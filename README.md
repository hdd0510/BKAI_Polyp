# Guide to run infer.py

This guide provides instructions on how to download the model checkpoint from Google Drive and save it to the `/kaggle/working/` directory in a Kaggle notebook.

## Prerequisites

- Ensure that the Kaggle notebook has internet access enabled.

## Instructions
```python
import requests
import os

# Replace YOUR_DRIVE_URL with the direct link to the Google Drive file
drive_url = 'YOUR_DRIVE_URL'
''''
https://drive.google.com/u/0/uc?id=1rE66914xj9HfNXFHGjtMxMq--Hbk3A69&export=download&confirm=t&uuid=2b4102a9\
    -5972-416b-97eb-88ba28ee326d&at=AB6BwCAGuaEHjfdCyfAwGaV0E-O9:1700047389408
'''
# Directory where the downloaded file will be saved
save_dir = '/kaggle/working/'

# Send a GET request to the drive_url
response = requests.get(drive_url)

# Write the content of the response to a file in the save_dir
with open(os.path.join(save_dir, 'model.pth'), 'wb') as f:
    f.write(response.content)
```
```python
!git clone https://github.com/hdd0510/BKAI_Polyp.git # clone my git repo
```
```python
!mkdir predicted_mask # make dir for mask prediction
```
```python
!python /kaggle/working/BKAI_Polyp/infer.py --checkpoint '/kaggle/working/model.pth' --test_dir '/kaggle/input/bkai-igh-neopolyp/test/test' --mask_dir '/kaggle/working/predicted_mask'

# parse args checkpoint, test_dir (please add data of competition), mask_dir

