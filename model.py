# Import required libraries
import os
import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.utils import np_utils
# Define paths to the dataset and output directory
data_dir = "C:/Users/yashj/OneDrive/Desktop/dataset"
output_dir = "./models"
# Load dataset
def load_dataset():
X = []
y = []
for subdir, dirs, files in os.walk(data_dir):
for file in files:
file_path = subdir + os.path.sep + file
label = subdir.split("/")[-1]
rate, data = wavfile.read(file_path)
X.append(data)
y.append(label)
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = np_utils.to_categorical(y)
return np.array(X), y
# Split dataset into training and testing sets
X, y = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size=0.2, random_state=42)
# Define model architecture
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', 
input_shape=(X_train.shape[1], 1)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', 
metrics=['accuracy'])
# Train model
model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), 
y_train, validation_data=(X_test.reshape(X_test.shape[0], 
X_test.shape[1], 1), y_test), epochs=50, batch_size=128)
# Evaluate model
scores = model.evaluate(X_test.reshape(X_test.shape[0], 
X_test.shape[1], 1), y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
# Save model
model.save(output_dir + "/audio_intrusion_detection_model.h5")
