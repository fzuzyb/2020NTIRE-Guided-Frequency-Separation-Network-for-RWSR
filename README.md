# 2020NTIRE-Guided-Frequency-Separation-Network-for-RWSR


## Quick Start 
###Testing
#### 1: 
git clone https://github.com/fzuzyb/2020NTIRE-Guided-Frequency-Separation-Network-for-RWSR.git
#### 2: 
Download the model form [BaiduNetDisk](https://pan.baidu.com/s/12tPz7ZPOewjMtu4TvqBKOA) access code: bqsv

There are different model in BaiduNetDisk: Normal(nb=9) and Large(nb=23). If you use the Large model please modifiy
the config file in .codes/run_options/test  test_SR_Track1.yml or test_SR_Track2.yml
#### 3: 
Place the testing low-resolution images (Track1 and Track2)  in ./dataset/Track1 and ./dataset/Track2
#### 4: 
Place the downloaded models in ./experiments/Track1/models and ./experiments/Track2/models
#### 5: 
cd ./codes/scripts

sh run_scripts_test_Track1.sh

sh run_scripts_test_Track2.sh


## If you find the code useful, please cite our paper:

@inproceedings{Zhou2020GuideFS,

        title={Guided Frequency Separation Network for Real-World Super-Resolution},
        
        author={Yuanbo Zhou and Wei Deng and Tong Tong and  others},
        
        booktitle={CVPR Workshops},
        
        year={2020},
        
}

## reference
## Acknowledgment
https://github.com/open-mmlab/mmsr

