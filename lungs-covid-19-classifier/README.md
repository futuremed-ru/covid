# Lungs COVID-19 Classifier

Training code for lungs classification on two classes: "COVID-19" and "Other".  
Used framework: [PyTorch](https://pytorch.org/).

## Data
Dataset is meant to has two classes: "COVID-19" and "Other".
It was created by combining samples from four sources:
1. [The famous github repo](https://github.com/ieee8023/covid-chestxray-dataset) with COVID-19 images. 
2. Images from [Italian database](https://www.sirm.org/category/senza-categoria/covid-19/) with COVID-19 cases.
3. Kaggle [chest X-ray pneumonia dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
4. [NIH ChestXRay-14 dataset](https://www.kaggle.com/nih-chest-xrays/data).

To get images with COVID-19 the following was done:  
Most of the images from Italian database had been already included into the github repo. But some of them didn't.  
So after combining first two sources we got the all available at the moment (7 April 2020) images with COVID-19 and couple images without it (with other pathology or "no finding", they had been used as "Other" class samples).  
One patient can have multiple images in that part of the dataset.

To get more images for "Other" class the following was done:
* Randomly picked 450 images from chest X-ray pneumonia dataset in a ballanced manner.  
Patiend IDs were not taken into account while picking iamges for that part of the dataset.  
Totaly were picked:  
150 images of 149 patients with no finding,  
150 images of 144 patients with viral pneumonia,  
150 images of 144 patients with bacterial pneumonia.  
* Randomly picked 450 images from NIH ChestXRay-14 dataset:  
30 images with every of 14 pathologies and another 30 images with "no finding" label.  
Images with one target pathology may contain other pathologies as well.  
So this part of the dataset is *almost* ballanced.  
The images were picked in a way, that sub-dataset may contain only one image of a certain patient.
In other words we had 450 unique patient images.

Next, we combined all the data. And resulting dataset stats were following:

| Label         | Images count | Patients count |
| ------------- |-------------:| --------------:|
| COVID-19      |          143 |             96 |
| Other         |          949 |            907 |
| Total         |         1092 |           1003 |

All the images were resized to 564x564.
Mean and standard deviation were calculated for the images in the dataset.

## Training
DenseNet-121 was choosed as a backbone for the model. We used pretrained on ChestXRay-14 model for weight initialization.  
Working with medical images it's absolutely crucial to make sure that different images of one patient won't get into training/validation/test sets.  
To address this issue and due to the scarsity of COVID-19 images, 10-fold cross-validation over *patients* was used for training.  
Data augmentations used for training:
* Random rotate (<15°),
* Random crop (512x512),
* Random intensity shift.  

Center crops used for evaluation.  
Calculated mean and std were used to standardize images after augmentation.  
The network was modified to produce two logits for the classes ("COVID-19" and "Other").
Weighted binary cross-entropy used as the loss function.
As we cross-validate over *patients*, the number of images for each of the two classes changes from one fold to another, so we calculated perclass weights for every fold on the fly.  
The network was trained using Adam optimizer with asmgrad. Other hyperparemeters can be found in `config.py` file.
Best on validation set by ROC AUC model was saved for each fold.  

## Results
Network's predictions was obrained as argmax for produced scores for each class ("COVID-19" and "Other").  
Resulting models formed an ensemble which is used for further analisys.
Models stats:  

| Fold # | Val AUC | Val Acc |
| -----: | ------: | ------: |
|      1 | 0.99896 |     1.0 |
|      2 |     1.0 | 0.88235 |
|      3 | 0.98802 |     1.0 |
|      4 |     1.0 |     1.0 |
|      5 | 0.99312 |     1.0 |
|      6 | 0.96506 | 0.72222 |
|      7 | 0.99894 |     0.9 |
|      8 |     1.0 |     1.0 |
|      9 |  0.9937 |     1.0 |
|     10 |     1.0 |     1.0 |
|   Mean | 0.99387 | 0.95046 |

For testing were used new frontal (PA or AP views) X-ray images from the github repo (the ones that were added from 7 to 22 April 2020).  
And required to ballance ("COVID-19" and "Other") classes number of images were added from unused in training patient's images randomly picked from ChestXRay-14 (as they was picked randomly, statistically most of them were with "no finding" label).  
All that images with corresponding labels form test set.  
Per label stats on the test set are in tables below.  

| Label              | Predicted "COVID-19" count | Predicted "Other" count | Total |
| ------------------ | -------------------------: | ----------------------: | ----: |
| COVID-19           | 57 | 7 | 64 |
| COVID-19, ARDS     | 4 | 0 | 4 |
| SARS               | 4 | 1 | 5 |
| No Fiding          | 0 | 1 | 1 |
| ChestXRay14 picked | 1 | 61 | 62 |
| Total              | 66 | 70 | 136 |

Summarising the stats above for "COVID-19" and "Other" classes we get:

| Label              | Predicted "COVID-19" count | Predicted "Other" count | Total |
| ------------------ | -------------------------: | ----------------------: | ------: |
| COVID-19           | 61 | 7 | 68 |
| Other              | 5 | 63 | 68 |
| Total              | 66 | 70 | 136 |

Common metrics for resulting "COVID-19" ensemble classifier:

| Metric                  | Value   |
| ----------------------- | ------: |
| Accuracy                | 0.91176 |
| Precision               | 0.92424 |
| Recall (Sensitivity)    | 0.89706 |
| Specificity             | 0.92647 |
| F1 score                | 0.91045 |

If these results convinced you that here we have a good Chest X-ray COVID-19 classifier, check out [this folder](https://github.com/futuremed-ru/covid/tree/master/performance-analysis) with deeper analisys of the classifier's performance.
