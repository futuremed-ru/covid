# Lungs COVID-19 Classifier Analysis

This repo contains code and description for performance analysis of [this COVID-19 classifier](https://github.com/futuremed-ru/covid/tree/master/lungs-covid-19-classifier).

## Performance analysis
In the repo above you can see very good performance metrics. Let's find out if they really indicative.

### ChestXRay-14
First, let's look at classifier's performance stats on the rest (unused in training) of [ChestXRay-14](https://www.kaggle.com/nih-chest-xrays/data) dataset:

| Label              | Predicted "COVID-19" count | Predicted "Other" count | Total |
| ------------------ | -------------------------: | ----------------------: | ----: |
| Atelectasis        | 16 | 6766 | 6782 |
| Cardiomegaly       | 14 | 1711 | 1725 |
| Consolidation      | 4 | 2058 | 2062 |
| Edema              | 12 | 1228 | 1240 |
| Effusion           | 3 | 6602 | 6605 |
| Emphysema          | 5 | 1102 | 1107 |
| Fibrosis           | 3 | 1016 | 1019 |
| Hernia             | 0 | 150 | 150 |
| Infiltration       | 36 | 7977 | 8013 |
| Mass               | 9 | 3741 | 3750 |
| Nodule             | 15 | 3324 | 3339 |
| Pleural_Thickening | 1 | 1442 | 1443 |
| Pneumonia          | 5 | 876 | 881 |
| Pneumothorax       | 2 | 2177 | 2179 |
| No Finding         | 221 | 54659 | 54880 |
| Total              | 319 | 83110 | 83429 |

Since in that dataset there's no COVID-19 cases, then the only thing we can claim is that our classifier has pretty good specificity (0.99235) *on this dataset*.  
Also you can see that there's no peak of false positives on such classes as "Pneumonia" and "Infiltration" - the ones which might have similar to COVID-19 X-ray picture.   
Does it mean that COVID-19 can be distinguished from other similar looking pathologies by an AI algorithm? 

### Proprietary dataset

To answer the posed question, let's have a look at the classifier's performance on *unseen* data.  
Used dataset has only "normal" and "abnormal" labels. There's no COVID-19 positive patient images in thit dataset.  

| Label    | Predicted "COVID-19" count | Predicted "Other" count | Total |
| -------- | -------------------------: | ----------------------: | ----: |
| Normal   | 90 | 415 | 505 |
| Abnormal | 2379 | 5167 | 7546 |
| Total    | 2469 | 5582 | 8051 |

Specificity drops down significantly (to 0.69333). The classifier doesn't seems to be so good now. What happend?

### "New github" dataset

Results on new images (7-22 Apr 2020) from [github repo with COVID-19 cases](https://github.com/ieee8023/covid-chestxray-dataset).
Part of the table from [the classifier's repo](https://gitlab.com/futuremed/lungs-covid-19).

| Label              | Predicted "COVID-19" count | Predicted "Other" count | Total |
| ------------------ | -------------------------: | ----------------------: | ----: |
| COVID-19           | 57 | 7 | 64 |
| COVID-19, ARDS     | 4 | 0 | 4 |
| SARS               | 4 | 1 | 5 |
| No Fiding          | 0 | 1 | 1 |
| Total              | 65 | 9 | 74 |

### Summary statistics

Combining all the stats on listed datasets for "COVID-19" and "Other" classes, we get:

| Label    | Predicted "COVID-19" count | Predicted "Other" count | Total |
| -------- | -------------------------: | ----------------------: | ----: |
| COVID-19 | 61 | 7 | 68 |
| Other    | 2788 | 88692‬ | 91480 |
| Total    | 2849 | 88699 | 91548 |

Common metrics:

| Metric                  | Value | Comment |
| ----------------------- | ----: | :------ |
| Accuracy                | 0.96947 | Classes are heavily unballanced, not indicative |
| Precision               | 0.02141 | Here it is, the classifier doesn't really know how COVID-19 is look like |
| Recall (Sensitivity)    | 0.89706 | Heavily affected by ChestXRay-14 results, not indicative |
| Specificity             | 0.96952 | Heavily affected by ChestXRay-14 results, not indicative |
| F1 score                | 0.04182 | Harmonic mean of precision and recall, indicative |

## Discussion

Now lets speak about intuition behind all these results.  
As it was mentioned, resulting precision shows, that the classifier doesn't able to distinguish COVID-19 spsecific patterns in the X-ray images (in fact, that there's nothing *specific* to COVID-19 manifestations in X-ray images).  
But what does the classifier learned then, and why it performs well on github repo and ChestXRay-14 data?  
The classifier learned how images from datasets picked for "Other" class look like.  
And it also learned that any pathological pattern or things like *arrows on images* means it's "COVID-19", *given that the image doesn't look like it's from "Other" datasets.*  
So, generaly, the classifier learned to distinguish *something pathological and not looking like "Other" images*.  

That's why it marked almost every third image as COVID-19 on our proprietary dataset (containing images that doesn't look similar to "Other" images).  
The classifier knows *some* difference between normal and abnormal images though. It marked as COVID-19 every 3rd abnormal and every 5th normal image.  

Despite strong data augmentation while training, careful patient-wise k-fold cross-validation and weighted loss function, the classifier failed to perform well.  

We encourage anyone interested to reproduce our experiment.  
Actually you don't need any proprietary data, you may just exclude one dataset from "Other" class, and use it as "unseen".

## Takeaways

We'd like to point out two main consequences:
* Any nural netrowk will always try to find *the easiest way* to solve the task.
* Look closely to the data on which the model performance is showed. Not bare numbers.








