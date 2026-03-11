# SWIGGY AI DATA SCIENCE PROJECT
# Single file version

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dot, Flatten, Input
import tensorflow as tf

# ----------------------------------------------------
# 1 CREATE SYNTHETIC DATASET
# ----------------------------------------------------

np.random.seed(42)

n = 2000

data = pd.DataFrame({
    "user_id": np.random.randint(1,300,n),
    "restaurant_id": np.random.randint(1,150,n),
    "rating": np.random.randint(1,6,n),
    "order_value": np.random.randint(100,800,n),
    "delivery_time": np.random.randint(15,60,n),
    "distance_km": np.random.uniform(0.5,10,n),
    "order_hour": np.random.randint(0,24,n)
})

print("Dataset Sample")
print(data.head())

# ----------------------------------------------------
# 2 RECOMMENDATION SYSTEM
# ----------------------------------------------------

def restaurant_recommendation():

    pivot = data.pivot_table(
        index='user_id',
        columns='restaurant_id',
        values='rating'
    ).fillna(0)

    similarity = cosine_similarity(pivot)

    user_index = 0

    scores = list(enumerate(similarity[user_index]))

    scores = sorted(scores,key=lambda x:x[1],reverse=True)[1:5]

    similar_users = [pivot.index[i[0]] for i in scores]

    recommendations = pivot.loc[similar_users].mean().sort_values(ascending=False)

    print("\nRecommended Restaurants:")
    print(recommendations.head(5))


restaurant_recommendation()

# ----------------------------------------------------
# 3 DELIVERY TIME PREDICTION
# ----------------------------------------------------

X = data[['distance_km','order_value','order_hour']]
y = data['delivery_time']

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2
)

model = RandomForestRegressor()

model.fit(X_train,y_train)

prediction = model.predict(X_test[:5])

print("\nPredicted Delivery Times")
print(prediction)

# ----------------------------------------------------
# 4 CUSTOMER SEGMENTATION
# ----------------------------------------------------

customer_features = data.groupby("user_id")[[
    "order_value",
    "delivery_time"
]].mean()

kmeans = KMeans(n_clusters=4)

customer_features["segment"] = kmeans.fit_predict(customer_features)

print("\nCustomer Segments")
print(customer_features.head())

# ----------------------------------------------------
# 5 DEMAND FORECASTING WITH LSTM
# ----------------------------------------------------

demand = data.groupby("order_hour").size().values

window = 3

X_lstm = []
y_lstm = []

for i in range(len(demand)-window):
    X_lstm.append(demand[i:i+window])
    y_lstm.append(demand[i+window])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

X_lstm = X_lstm.reshape((X_lstm.shape[0],X_lstm.shape[1],1))

lstm_model = Sequential()

lstm_model.add(
    LSTM(32,input_shape=(window,1))
)

lstm_model.add(Dense(1))

lstm_model.compile(
    optimizer='adam',
    loss='mse'
)

lstm_model.fit(
    X_lstm,
    y_lstm,
    epochs=10,
    verbose=1
)

future = lstm_model.predict(X_lstm[:1])

print("\nForecasted Future Demand")
print(future)

# ----------------------------------------------------
# 6 DEEP LEARNING RECOMMENDER
# ----------------------------------------------------

user_ids = data['user_id'].astype("category").cat.codes.values
restaurant_ids = data['restaurant_id'].astype("category").cat.codes.values
ratings = data['rating'].values

num_users = data['user_id'].nunique()
num_restaurants = data['restaurant_id'].nunique()

user_input = Input(shape=(1,))
restaurant_input = Input(shape=(1,))

user_embed = Embedding(num_users,50)(user_input)
restaurant_embed = Embedding(num_restaurants,50)(restaurant_input)

dot = Dot(axes=2)([user_embed,restaurant_embed])

output = Flatten()(dot)

dl_model = tf.keras.Model(
    [user_input,restaurant_input],
    output
)

dl_model.compile(
    optimizer='adam',
    loss='mse'
)

dl_model.fit(
    [user_ids,restaurant_ids],
    ratings,
    epochs=5,
    batch_size=64
)

print("\nDeep Learning Recommendation Model Trained")

# ----------------------------------------------------
# 7 BASIC ANALYTICS
# ----------------------------------------------------

print("\nOrders Per Hour")
print(data.groupby("order_hour").size())

print("\nAverage Delivery Time by Hour")
print(data.groupby("order_hour")["delivery_time"].mean())

print("\nProject Execution Completed Successfully")