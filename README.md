# Overview
This project involves building a branched neural network to help an HR department predict two key outcomes: (1) whether employees are likely to leave the company (attrition), and (2) which department an employee might be better suited for. The neural network model uses multiple input features and branches into two separate output layers to handle both predictions simultaneously. The task is divided into three main parts: preprocessing the data, building and training the model, and summarizing the results.

# Repository Structure
neural-network-challenge-2/
│
├── data/
│   ├── employee_data.csv
│
├── notebooks/
│   ├── attrition.ipynb
│
├── README.md
│
└── results/
    ├── model_summary.txt
    ├── department_accuracy.txt
    ├── attrition_accuracy.txt

# Instructions
## Part 1: Preprocessing
Objective: Prepare the data for use in the neural network model.

Import Data:

Load the employee data and inspect the first five rows to understand the structure.
Explore Unique Values:

Determine the number of unique values in each column to identify potential categorical features.
Target and Features Selection:

Create the target dataset y_df, which includes the attrition and department columns.
Select at least 10 feature columns (excluding attrition and department) to create the feature dataset X_df.
Data Splitting:

Split the data into training and testing sets using train_test_split.
Convert to Numeric Data:

Convert categorical columns in X_df to numeric data using appropriate encoding techniques such as OneHotEncoder or LabelEncoder.
Ensure that encoders are fitted on the training data and then applied to both training and testing datasets.
Feature Scaling:

Use StandardScaler to scale the feature data. Fit the scaler on the training data and transform both training and testing datasets.
Encode Target Columns:

Create a OneHotEncoder for the department column, fit it on the training data, and transform both training and testing datasets.
Similarly, create a OneHotEncoder for the attrition column and apply it to the training and testing datasets.
Output: Preprocessed training and testing datasets for both features (X) and targets (y).

## Part 2: Create, Compile, and Train the Model
Objective: Build and train a branched neural network model using TensorFlow's Keras API.

Determine Input Features:

Find the number of columns in the training data to determine the number of input features for the neural network.
Create Input Layer:

Create an input layer that connects to the shared layers of the neural network. Do NOT use a sequential model since the network will branch into two output layers.
Shared Layers:

Add at least two shared hidden layers between the input and the output branches.
Branch for Department Prediction:

Create a branch that handles the prediction of the department target. This branch should include at least one hidden layer and an output layer.
Branch for Attrition Prediction:

Create another branch for predicting attrition. This branch should also include at least one hidden layer and an output layer.
Create and Compile the Model:

Build the model by combining the branches and the shared layers.
Compile the model using appropriate loss functions, optimizers, and evaluation metrics (e.g., binary_crossentropy for attrition and categorical_crossentropy for department).
Model Summary:

Summarize the model structure to review the layers and parameters.
Train the Model:

Train the model using the preprocessed training data.
Evaluate the Model:

Evaluate the model's performance on the testing data and print the accuracy for both department and attrition predictions.
Output: Model summary and accuracy metrics for both department and attrition predictions.

## Part 3: Summary
Objective: Reflect on the model's performance and potential improvements.

Accuracy as a Metric:

Answer whether accuracy is the best metric to evaluate this model. Consider factors such as class imbalance, and suggest alternative metrics if necessary.
Activation Functions:

Discuss the activation functions chosen for the output layers (e.g., softmax for multi-class classification in department prediction and sigmoid for binary classification in attrition prediction). Explain why these choices were made.
Model Improvement:

Suggest a few ways to improve the model. Potential improvements might include adding more hidden layers, experimenting with different activation functions, or tuning hyperparameters such as the learning rate.
Output: Written answers to the summary questions in the markdown cells of your notebook.

# Conclusion
This project demonstrates how to preprocess data and build a branched neural network model to predict employee attrition and department suitability. The dual-output neural network allows HR departments to assess employee retention risks and recommend better-fitting roles within the company. By evaluating the model and reflecting on potential improvements, this analysis provides a foundation for building more accurate and practical predictive models in HR analytics.