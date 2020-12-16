# Taxi-Fare-Prediction
Regression for NYC Predicting Taxi Fare  with Keras based on pick-up coordinates, drop-off coordinates, pick-up time and the number of passengers. RMSE score of 3.67 (Linear model reaches 5.7 in comparison) www.kaggle.com/c/new-york-city-taxi-fare-prediction

Simple neural network architecture on Keras of 2 dense layers with size 50, with PReLu activation function, dropouts and layer normalizations. Reduction of the learning rate after 10 epochs without progress. The loss function here is MSE.

The closer a data point lies to the diagonal line, the better the prediction.

![Test Image 1](https://github.com/fallintoplace/Predicting-Taxi-Fare/blob/master/prediction_graph.png)
![Test Image 2](https://github.com/fallintoplace/Predicting-Taxi-Fare/blob/master/loss_graph.png)

