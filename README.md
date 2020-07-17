# Predicting-Taxi-Fare
Regression for NYC Predicting Taxi Fare  with Keras based on coordinates, pick-up time and the number of passengers. RMSE score of 3.67 (Top 55% of the leaderboard &amp; linear model reaches 5.7 in comparison) www.kaggle.com/c/new-york-city-taxi-fare-prediction

Simple neural network architecture on Keras of 2 dense layers with size 50, with PReLu activation function, dropouts, layer normalizations and regularizations. Halves the learning rate after 5 epochs without progress. The loss function here is MSE.

![Test Image 1](https://github.com/fallintoplace/Predicting-Taxi-Fare/blob/master/prediction_graph.png)
![Test Image 2](https://github.com/fallintoplace/Predicting-Taxi-Fare/blob/master/loss_graph.png)

