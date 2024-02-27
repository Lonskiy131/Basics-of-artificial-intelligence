import numpy as np
from sklearn import preprocessing

input_data = np.array([[-3.3, -1.6, 6.1],
                       [2.4, -1.2, 4.3],
                       [-3.2, 5.5, -6.1],
                       [-4.4, 1.4, -1.2]])

data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print(f"Binarized data: \n{data_binarized}")

print("\nBefore:")
print("Mean = ", input_data.mean(axis=0))
print("Std deviation = ", input_data.std(axis=0))
data_scaled = preprocessing.scale(input_data)
print("\nAfter:")
print("Mean = ", data_scaled.mean(axis=0))
print("Std deviation = ", data_scaled.std(axis=0))

data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin max scaled data:\n", data_scaled_minmax)

data_normalized_l1 = preprocessing.normalize(input_data, norm="l1")
data_normalized_l2 = preprocessing.normalize(input_data, norm="l2")
print("\nl1 normalized data:\n", data_normalized_l1)
print("l2 normalized data:\n", data_normalized_l2)
