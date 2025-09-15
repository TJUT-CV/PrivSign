# PirvSign
This repo holds codes of the paper: Privacy-Preserving Continuous Sign Language Recognition. (TMM 2025).

This repo is based on [VAC (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html) and [CorrNet (CVPR 2023)] (https://arxiv.org/abs/2303.03202). Many thanks for their great work!

## Perequisites

This project is implemented in Pytorch (>1.8). Thus please install Pytorch first.

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.

- sclite [[kaldi-asr/kaldi]](https://github.com/kaldi-asr/kaldi), install kaldi tool to get sclite for evaluation. After installation, create a soft link toward the sclite: 
  `mkdir ./software`
  `ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite`

- [SeanNaren/warp-ctc](https://github.com/SeanNaren/warp-ctc) for ctc supervision.

- You can install other required modules by conducting 
   `pip install -r requirements.txt`

## Implementation
The implementation for the FCS (line 18) is given in the function images_to_batch in line 97 in [./slr_network.py](https://github.com/TJUT-CV/PrivSign/blob/main/slr_network.py).  

### PHOENIX2014 dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). Our experiments based on phoenix-2014.v3.tar.gz.

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/phoenix2014-release ./dataset/phoenix2014`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess.py --process-image --multiprocessing
   ```

### PHOENIX2014-T dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/PHOENIX-2014-T-release-v3/PHOENIX-2014-T ./dataset/phoenix2014-T`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess-T.py --process-image --multiprocessing
   ```

### CSL dataset

1. Request the CSL Dataset from this website [[download link]](https://ustc-slr.github.io/openresources/cslr-dataset-2015/index.html)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET ./dataset/CSL`

3. The original image sequence is 1280x720, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess-CSL.py --process-image --multiprocessing
   ``` 

### CSL-Daily dataset

1. Request the CSL-Daily Dataset from this website [[download link]](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET ./dataset/CSL-Daily`

3. The original image sequence is 1280x720, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess-CSL-Daily.py --process-image --multiprocessing

## Inference
​	To evaluate the pretrained model, choose the dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in ./config/baseline.yaml first, and run the command below：   
`python main.py --device your_device --load-weights path_to_weight.pt --phase test`

Pretrained models  [[Google Drive]](https://drive.google.com/drive/folders/1wLfQsMBC0RsRISkrQu5nRf9bEGqlUMP3?usp=sharing)
### PHOENIX2014 dataset

| Backbone | Dev WER  | Test WER  |                                            
| -------- | ---------- | ----------- | 
| ResNet18 | 18.4%      | 19.2%       | 

### PHOENIX2014-T dataset

| Backbone | Dev WER  | Test WER  |                                             
| -------- | ---------- | ----------- | 
| ResNet18 | 18.5%      | 20.2%       | 

### CSL-Daily dataset

| Backbone | Dev WER  | Test WER  | 
| -------- | ---------- | ----------- | 
| ResNet18 | 28.0%      | 27.7%       | 

