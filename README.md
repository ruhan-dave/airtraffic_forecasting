# airtraffic_forecasting


# churn

Slides: https://docs.google.com/presentation/d/13Gn8FUjrlsuypsWgiI9v6VZoiQY8acOq8UMfLgTDu8E/edit?usp=sharing

Demo App with Streamlit: https://sf-airtraffic-forecasting.streamlit.app/


1. Problem Statement

Imagine how airports decide how many flights there will be? Or how many food supplies to get ready for the many indoor restaurants? Or how to charge? It would be a nightmare without a way of knowing how many customers will be taking flights. There needs to be a solution for forecasting air traffic demand. This project does just that for one of the busiest airports —— the San Francisco International Airport! 

2. Data Science/Machine Learning Project Steps

#### Data Collection

Fetched data directly from the SF International Airport website. There also exists a Kaggle dataset with the same data. The years ran from 1996 to 2023, and we’d predict the values for 2024. 

#### Data Exploration & Preprocessing

There were many plots drawn, including breakdowns of different terminals, flights over time by airlines, months, years, flight destinations, gates, and scatter plots. I did bar plots and box plots for comparing across different subgroups of data. I even found mislabeled data with a few airlines that would otherwise skew with the modeling! For more details, please view slides, linked above. 

#### Feature Engineering & Selection

For the time-series model, I used ARIMA with seasonality and trend decomposition. These steps were to make sure that 
For the hyper-parameters, I did some fine-tuning iteratively to see the best combinations that yielded the lowest MSE. 

For the traditional machine learning approach, I used the XGBoost regression model and did grid search for the hyper-parameters, which were proven successful as well. 

#### Model Selection & Training

Split data into training, validation, and test sets.
Trained the selected model, an XGBoost classifier and tuned hyperparameters for optimal performance.

#### Model Evaluation & Deployment

As said, the MSE was used as a loss function. However, MAPE (average percentage error) makes more intuitive sense for non-data-experts. 

Evaluated the final model's performance on a held-out test set.
Deployed the model on Streamlit by first designing an intuitive Streamlit interface and then hosting in with private app keys and secrets.

3. Outcomes Achieved

#### Quantifiable Results:

Built a streamlit web application that users can input information about the airline, destination, year, month, gate, flight type, operation type, and more. With the time series deployed on AWS S3, the user can then check the demand for the specific month and year! 

#### Impact:

This highly accurate time-series forecasting project significantly helps with planning and traffic management. Moreover, this can also help with setting plane ticket prices depending on demand. 
