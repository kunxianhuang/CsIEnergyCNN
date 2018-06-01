# CsIEnergyCNN
Correct gamma-ray energy @ CsI Array by using CNN method 

Explanation slide at PyCon 2018: https://goo.gl/7AUoBL

Training Data download: https://figshare.com/s/cd24b4ab2126fc78b79b \
You can download them by push button of download and unzip, then put it into directory of train_data/

Testing Data download: https://figshare.com/s/cc8c5a8df5e5ec5b910c \
You can download them by push button of download and unzip, then put it into directory of test_data/


## Usage
### Training 
After downloading the training data and saving them into train_data, you can use the below command to train \
our model the default step is 30,000. It will take about 2 days.
``python CsIArray_CNN_train.py ``

The trained model will be saved into directory of save_model/

### Applying
You can download the testing data which are CsI array energies with incident of mono-energy gamma-ray.
