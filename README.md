# match-plans

## Problem Description

In ROME (Renewal Optimization Engine), a fascet of the 'cold start problem' is that we receive a medical plan (characterized by plan_admin_name or plan_name) and in order to complete current + renewal total cost, we must match those plans to plans in the Ideon dataset of plans available in the clients zip code. As of now, this manual matching process is time-consuming and a roadblock to potentially automating the configuration steps in ROME for small groups.

## Objective

The objective of this project is to automate the matching process by developing a machine learning model that can accurately link a given medical plan name from a clients census to the correct plan in the ideon dataset. This is a data linkeage problem, one that involves text cleaning, text similarity, and some feature engineering.

## Proposed Solution

We propose building a dataset that combines / builds features extracted from the carrier and plan_admin_name or plan_name string from a census with the features from the Ideon dataset of medical plans. This dataset will then be used to train a machine learning model that can predict the correct plan from the dataset for a given plan name string and carrier string. The target variable in this dataset will be a binary variable indicating a match (1 to represent positive) or not a match (0 to represent a negative).

### Feature Engineering

- **String-based Features**: Compute string similarity measures such as edit distance and token overlap between the input plan name and the plan names in our dataset.
- **Categorical and Numerical Features**: Use existing features such as deductible, coinsurance, and metallic level from the Ideon dataset.
  - **Plan Type**: extracting the plan type from the census plan string ['PPO','HMO']
  - **Metallic Level**: find and extract the metallic level; ['Gold','Bronze','Silver','Platinum']
  - **Network**: this one might be more difficult, but finding a way to extract the network, as the network is oftentimes in the plan string name, especially if a national carrier (UHC, Aetna, Anthem, BlueShield)
  - **HDHP/HSA**: oftentimes, a plan will have HDHP or HSA in the plan name, which can helps us build a feature that could be meaningful to the model
  - **Plan Characteristics**:
    1. If there are any numbers in the plan (not including Gold 80 for Kaiser as an example), then we should extract those numbers and match them with the numbers that are located in the plan benefits
    2. Coinsurance / In Network Individual Deductible / PCP Copay are usually the values
  - **Carrier Specific Features**:
    1. California Choice: the plans are recognized by an "A" or "B" or an alphabetical letter that identifies the plan. We should be able to determine if the carrier is California Choice and then extract the letter and store it as a feature.
    2. UnitedHealthcare: we oftentimes get an issuer_plan_code either in the format of XXXX or XX-XX. We should be able to deterine if UnitedHealthcare and then extract this code if it matches these patters.
    3. Anthem: we also get an issuer_plan_code with Anthem in the format of XXXX. We should be able to determine if Anthem and then extract this code if it matches the pattern
- **Similarity Score**: to add a feature to the model I could produce a "similarity" score between a bunch of things... 
  1. I'm thinking one way to do this is to create a feature that takes everything and jams it into vector space...

### Model

We propose using XGBoost for this project. It generall has a "throw anything at it, it will work" proven track record on a wide range of machine learning problems, including binary classification tasks similar to this one.

#### Why XGBoost:
- XGBoost stands for eXtreme Gradient Boosting, and it is an efficient and scalable implementation of gradient boosting.
- One of the key benefits of XGBoost is its ability to handle missing data, which can be useful in our case if there are any missing values in the dataset. 
  * an example here: we are missing plan_type from the plan_admin_name, so it can handle that missing value
- XGBoost also allows for regularization, which helps to prevent overfitting and improve the model's generalization performance.
- Another advantage of XGBoost is its flexibility, as it supports various objective functions and evaluation criteria, making it a versatile choice for a range of problems.

To find the optimal hyperparameters for our XGBoost model, we will use GridSearchCV. This method performs an exhaustive search over a specified parameter grid, allowing us to find the combination of parameters that yields the best performance.

#### How GridSearchCV Works:
- We will define a grid of hyperparameter values and pass it to GridSearchCV, along with the XGBoost model and the training data.
- GridSearchCV will then train the model with each combination of hyperparameters and evaluate its performance using cross-validation.
- Once the search is completed, we can access the best parameters found by GridSearchCV and use them to train our final model.

By combining XGBoost with GridSearchCV, we will be able to leverage the strengths of the XGBoost algorithm while also ensuring that we are using the optimal set of hyperparameters for our specific problem.


### Scoring

For evaluation: we will see which scoring we need to implement: f1, roc_auc, accuracy, etc.
#### Returning Confidence Number:

- For each instance that the model predicts, it will also return the predicted probability of that instance belonging to the positive class (i.e., being a match).
- This probability score can be interpreted as the model's confidence in its prediction.
- For example, if the model predicts that a specific instance is a match and returns a probability score of 0.85, it means that the model is 85% confident in its prediction.

#### Why Accuracy Estimation is Useful:

- The accuracy estimation provides additional information that can help in interpreting the model's predictions.
- It can be especially useful in cases where the prediction is close to the decision boundary, as the confidence score can help in deciding how much trust to place in the prediction.
- This approach allows for a more nuanced interpretation of the model's predictions, taking into account the uncertainty associated with each prediction.

## Project Structure

- `/data`: Directory containing the dataset used for training
- `/notebooks`: Jupyter notebooks for exploratory data analysis and model development.
- `/src`: Python scripts for data preprocessing, feature engineering, and model training.
- `/models`: The trained model
