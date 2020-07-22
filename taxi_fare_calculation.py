import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn import preprocessing
from matplotlib import pyplot as plt
from matplotlib import dates as md

column_names = ['fare_amount','pickup_datetime', 'pickup_longitude', 'pickup_latitude',
                'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

"""
PREPARING THE DATASET

"""

def process(file_url):
    dataset = pd.read_csv(file_url, nrows=50000)    

    dataset = dataset.dropna()
    dataset = dataset[dataset['dropoff_latitude'] != 0]
    dataset = dataset[dataset['dropoff_longitude'] != 0]
    dataset = dataset[dataset['pickup_latitude'] != 0]
    dataset = dataset[dataset['pickup_longitude'] != 0]
    
    dataset['diff_longtitude'] = dataset.apply(lambda x: 
                    abs(x['pickup_longitude'] - x['dropoff_longitude']), axis=1)
    dataset['diff_latitude'] = dataset.apply(lambda x: 
                    abs(x['pickup_latitude'] - x['dropoff_latitude']), axis=1)
    
    dataset['pickup_datetime'] = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S %Z") 
                                   for x in dataset['pickup_datetime']]
    dataset['hour'] = dataset['pickup_datetime'].apply(lambda x: x.hour)
    dataset['year'] = dataset['pickup_datetime'].apply(lambda x: x.year)
    dataset['day'] = dataset['pickup_datetime'].apply(lambda x: x.dayofyear)
    dataset['weekday'] = dataset['pickup_datetime'].apply(lambda x: x.weekday())
    dataset['week'] = dataset['pickup_datetime'].apply(lambda x: x.week)
   
    dataset.pop('pickup_datetime')
    return dataset


dataset = process("C:\\Users\\Minh\\Downloads\\new-york-city-taxi-fare-prediction\\train.csv")
dataset.pop('key')

dataset = dataset.sample(frac=1).reset_index(drop=True)

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_labels = train_dataset.pop('fare_amount')
test_labels = test_dataset.pop('fare_amount')

final = process("C:\\Users\\Minh\\Downloads\\new-york-city-taxi-fare-prediction\\test.csv")
final_key = final.pop('key')


train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
print(train_stats.head())

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
final = norm(final)
test_dataset = norm(test_dataset)
train_dataset = norm(train_dataset)


"""
TWO DENSELY PRELU ACTIVATED CONNECTED LAYERS WITH DROPOUTS AND LAYER NORMALIZATIONS

"""

def build_model():
    model = keras.Sequential([
        layers.Dense(50, 
                     input_shape = [len(train_dataset.keys())],
                     ),
        layers.PReLU(alpha_initializer=tf.initializers.constant(0.25)),
        layers.LayerNormalization(),
        layers.Dropout(rate = 0.5),
        layers.Dense(50),
        layers.PReLU(alpha_initializer=tf.initializers.constant(0.25)),
        layers.LayerNormalization(),
        layers.Dropout(rate = 0.5),
        layers.Dense(1)
        ])

    model.compile(loss = 'mse',
                optimizer = tf.keras.optimizers.Adam(0.0005),
                )
    return model

model = build_model()
model.summary()


"""
FITTING AND PLOTTING THE LOSS VALUE GRAPH

"""

history = model.fit(
    train_dataset, train_labels,
    epochs = 1000, batch_size = 128, validation_split = 0.2, verbose = 1,
    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                            patience=10, 
                                            verbose=1, 
                                            factor=0.6, 
                                            min_lr=0.00005)],
)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

"""
PLOTTING THE PREDICTION GRAPH

"""
train_predictions = model.predict(train_dataset).flatten()
plt.axes(aspect = 'equal')
plt.scatter(train_labels, train_predictions, s=1, color="b", label="Training")
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 100]
plt.xlim(lims)
plt.ylim(lims)
plt.legend()
plt.grid(True)
plt.plot(lims, lims)



test_predictions = model.predict(test_dataset).flatten()
plt.axes(aspect = 'equal')
plt.scatter(test_labels, test_predictions, s=1, color="r", label="Test")
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 100]
plt.xlim(lims)
plt.ylim(lims)
plt.legend()
plt.grid(True)
plt.plot(lims, lims)

"""
PRINTING THE RESULT

"""




result = pd.DataFrame()
result['key'] = final_key
result['fare_amount']=model.predict(final).flatten()
result['fare_amount']=result['fare_amount'].apply(lambda x: 1 if (x<1) else x)
pd.DataFrame(result).to_csv("C:\\Users\\Minh\\Downloads\\new-york-city-taxi-fare-prediction\\result.csv", index = False)





