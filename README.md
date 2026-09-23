# Weather Prediction Using Machine Learning

Weather Prediction Using Machine Learning is a machine learning project developed to predict different weather conditions using historical atmospheric data. Weather forecasting is important in areas such as farming, transportation, business, and everyday planning, but accurately predicting weather can be challenging because atmospheric conditions are complex and constantly changing.

In this project, weather prediction is treated as a classification problem. A dataset containing more than **96,000 historical weather records** is used, with atmospheric features such as temperature, humidity, pressure, wind speed, visibility, and precipitation-related information. The data is cleaned and preprocessed by handling missing values, removing duplicate records, and converting categorical variables into numerical values.

Multiple machine learning models are trained and compared to determine how well they can classify different weather conditions. The models used in the project are **Gradient Boosting, XGBoost, Extra Trees, and Random Forest**. The dataset is divided into **80% training data and 20% testing data**, with stratified sampling used to maintain the distribution of weather classes.

Among the tested models, **Random Forest achieved the highest accuracy of 89.67%**, followed by Extra Trees with 88.23%, XGBoost with 80.77%, and Gradient Boosting with 74.49%. This comparison demonstrates how different ensemble learning techniques perform on the same weather classification task.

The project also includes a **Streamlit-based web application** that provides an interactive interface for weather prediction. Users can enter atmospheric parameters such as temperature, humidity, wind speed, pressure, and visibility. The application can also integrate with an online weather API to obtain current weather information based on a selected location and use those parameters for prediction.

Overall, this project demonstrates the use of machine learning and ensemble methods for weather classification using historical atmospheric data. It also shows how a trained machine learning model can be integrated into a simple web application to make predictions through an interactive interface.
