# Heart Attack Background
Conducting simple EDA and creating a model to assess the likelihood of a death by heart failure event.
This can be used to help hospitals in assessing the severity of patients with cardiovascular diseases.

# AirBnB Background
Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present a more unique, personalized way of experiencing the world. Today, Airbnb became one of a kind service that is used and recognized by the whole world. Data analysis on millions of listings provided through Airbnb is a crucial factor for the company. These millions of listings generate a lot of data - data that can be analyzed and used for security, business decisions, understanding of customers' and providers' (hosts) behavior and performance on the platform, guiding marketing initiatives, implementation of innovative additional services and much more.


## Data Source
This dataset was collected from [Kaggle](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data). 
This dataset has 299 observations and 13 variables.
After changing the datetupe in the dataset, this is what we have:

| Variable| Datatype | Description |
| ----------| ------------- | -------- |
| age | float64 | The patient's age |
| anaemia | category | Decrease of red blood cells or hemoglobin (boolean) |
| creatinine_phosphokinase | int64 | Level of the CPK enzyme in the blood (mcg/L) |
| diabetes | category | If the patient has diabetes (boolean) |
| ejection_fraction | int64 | Percentage of blood leaving the heart at each contraction (percentage) |
| high_blood_pressure | category | If the patient has hypertension (boolean) |
| platelets | float64 | Platelets in the blood (kiloplatelets/mL) |
| serum_creatinine | float64 | Level of serum creatinine in the blood (mg/dL) |
| serum_sodium | int64 | Level of serum sodium in the blood (mEq/L)|
| sex | category | Woman or man (binary) |
| smoking | category |If the patient smokes or not (boolean) |
| time | int64 | Follow-up period (days) |
| DEATH_EVENT | category | If the patient deceased during the follow-up period (boolean) |


## Medical Questions:
The following Medical Questions were thought of:

- [x] How many patients have diabetes?
- [x] How many patients have anaemia?
- [x] How many patients have high blood pressure?
- [x] How many patients are smokers?
- [x] How many patients passed away as a result?



# Learning Outcomes

The purpose of this analysis for myself is to: 
- [ ] analyse the data in 2 hours
- [ ] normalise the data
- [ ] split the data into training and test set
- [ ] run a decision tree and look at confusion matrix 


## Data preparation


## EDA for
[diabetes_graph](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)

[anaemia_graph](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)

[sex_graph](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)

[HB_graph](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)

[smoking_graph](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)

[death_graph](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)



## EDA for

## Data Preparation for Decision Tree
The data was normalised in numerical data. 

The data was split into 70%/30% for training and test set.

| Training Set | Test Set |
| ----------| ------------- |
| 70% | 30% | 
| age | float64 | 

Number of observations in each class before oversampling (training data):
| Variable | Current Data |
| ----------| ------------- |
| Lived | 144| 
| Death | 65 | 

Number of observations in each class AFTER oversampling (training data):
| Variable | Current Data |
| ----------| ------------- |
| Lived | 144| 
| Death | 144 | 

