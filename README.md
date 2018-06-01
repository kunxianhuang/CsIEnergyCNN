# CsIEnergyCNN
Correct gamma-ray energy @ CsI Array by using CNN method 

Explanation slide at PyCon 2018: https://goo.gl/7AUoBL

Training Data download: https://figshare.com/s/cd24b4ab2126fc78b79b \
You can download them by push button of download and unzip, then put it into directory of train_data/

Testing Data download: https://figshare.com/s/cc8c5a8df5e5ec5b910c \
You can download them by push button of download and unzip, then put it into directory of test_data/

## CNN architecture

#![CNN model](https://github.com/kunxianhuang/CsIEnergyCNN/blob/master/plots/CNN_structure.jpg "CsI Array CNN model")


## Usage

### Training 
After downloading the training data and saving them into train_data, you can use the below command to train \
our model the default step is 30,000. It will take about 2 days. \
``python3 CsIArray_CNN_train.py ``

The trained model will be saved into directory of save_model/

### Applying
You can download the testing data which are CsI array energies with incident of mono-energy gamma-ray. \
The mono-energy samples can be used to compare the corrected energy distribution with ones that is not applied correction method. The below command line shows how to apply trained model to the test data. \
``python3 CsIArray_CNN_train.py --test_file=[testfilename] --test_outfile=[outputfilename]``

``testfilename: read in file name of CsI array energy test data, which is in test_data/ ``
``outputfilename:  write out filename under directory of test_result/ ``

