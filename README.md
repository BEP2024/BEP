# A Bias Extraction and Penalty Method for Robust Visual Question Answering

This repo contains the PyTorch code release for the paper A Bias Extraction and Penalty Method for Robust Visual Question Answering.
The code will be organized and released after the paper is accepted. Currently, the dataset download procedure has been provided.


### Data Setup

Download UpDn features from [google drive](https://drive.google.com/drive/folders/111ipuYC0BeprYZhHXLzkRGeYAHcTT0WR?usp=sharing), which is the link from [this repo](https://github.com/GeraldHan/GGE), into ``/data/detection_features`` folder

Download questions/answers for VQA v2 and VQA-CP2 by executing ``bash tools/download.sh``

Preprocess process the data with bash ``tools/process.sh``


