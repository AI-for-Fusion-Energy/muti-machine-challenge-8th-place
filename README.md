# muti-machine-challenge-8th-place
This repository contains the code and description of our solution as well as the report and the final presentation slide.

AI-for-Fusion-Energy:The objective of this challenge is to develop a disruption prediction model that can be applied universally, using J-TEXT and HL-2A as the current devices and C-Mod as the future device.For more information about this challenge go to [AI-for-Fusion-Energy](https://zindi.africa/competitions/multi-machine-disruption-prediction-challenge/).

## Requirements

• python==3.9.7

• torch==2.0.0

• numpy==1.26.1

• tsai==0.3.7

• scikit-learn==1.2.2

You can install all the required packages with the following pip command:

```pip install torch==2.0.0 numpy==1.26.1 tsai==0.3.7 scikit-learn==1.2.2 ```

## Proposed Solution

After preprocessing the data, we used two models to train the processed data sets respectively. These two models are based on LSTM and MLP respectively. After the two models have trained a good model respectively, we ensemble the two models. The final integrated effect is slightly better than the best of the two models, which verifies that our ensemble is meaningful.

### Data Processing

The original data given in this challenge cannot be directly trained. We need to perform data preprocessing before using it to train the model. Regarding data preprocessing, we did the following work: First, according to the official signal correspondence table on the three machines, we extracted the signals in HL-2A and J-TEXT according to the signal names of C-MOD in the table. , name all the data in HL-2A and J-TEXT according to the corresponding C-MOD signal names in the table so that other researchers can use these data to train the model. Then, we align all signals according to their start time by comparing the "start_time" attribute in the properties of each signal. Then, we downsample all signals with a sampling rate of 5000. The purpose of this is to align all the data so that they have the same length for subsequent processing. Finally, we uniformly handle all outliers to ensure that no problems occur during subsequent training.For label division, we use different division methods for the HL-2A and J-TEXT data sets respectively. In HL-2A, we represent the first 100 time steps of the rupture point as 1, and other time steps as 0. In the J-TEXT data set we represent the first 40 time steps of the rupture point as 1 and other time steps as 0.

### LSTM
Because in the data set processed so far, only the lengths of the signals of each shot are the same, and the lengths of the signals of each shot are not the same. Therefore, before model training, the data was also processed with variable step size sliding window: in most time steps before the front rupture point of the gun that did not rupture or ruptured, a relatively large sliding window step size was used. A small step size is used to divide the sliding window in a relatively short time step before the point. The model can achieve good results by using the classic two-layer LSTM model for training.Among them, we save the model of each epoch, evaluate the model based on the verification loss of the model and analyze the model's predicted waveform for each shot, and finally upload it to the website to judge the true quality of the model.

### gMLP
The gMLP model was published by the Brain Team in Google Research in 2021. Papers on this model can be found [gMLP](https://arxiv.org/pdf/2105.08050.pdf).
The data preprocessing method using the gMLP model is basically consistent with the idea of the LSTM model. What is different from the LSTM model is that in the selection of step sizes for most time steps before the rupture point of the non-ruptured cannon and the ruptured cannon, the gMLP model Instead of using a fixed step size, a random step size is used. Moreover, the gMLP model chooses the window size to be 80, which is also different from the LSTM model.We use the tsai library to simplify our model training process. Through this python library, we can directly call the gMLP model to train the processed data. You can go to [tsai](https://github.com/timeseriesAI/tsai) if you also want to try this library.

### Ensemble
After we got two models with good results, we used the GBDT method to integrate our models, and then got our final result.Gradient Boosting Decision Trees (GBDT) is an ensemble learning method designed to build a powerful predictive model. The method involves iteratively training decision trees, with each tree focusing on correcting the prediction errors of the preceding one. In each iteration, the model minimizes the residuals through gradient descent, gradually enhancing overall performance. Ultimately, the combination of multiple weak learners (decision trees) forms a robust ensemble model applicable to both regression and classification problems. GBDT excels in handling complex data and nonlinear relationships, making it widely employed in practical applications.
