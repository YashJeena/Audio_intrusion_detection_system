# Detection of Intrusion
from keras.models import load_model
from scipy.io import wavfile
# Load saved model
model = load_model("C:/Users/yashj/OneDrive/Desktop/ML audio 
Intrusion Detection/models" + "/audio_intrusion_detection_model.h5")
# Load new audio data
rate, data = 
wavfile.read("C:/Users/yashj/OneDrive/Desktop/testing_audio1.wav")
# Reshape audio data to match model input shape
data = data.reshape(1, data.shape[0], 1)
# Make prediction
prediction = model.predict(data)
print(prediction)
# Decode prediction
intrusion_detected = prediction[0] > 0.5
if intrusion_detected:
print("Intrusion detected!")
else:
print("No intrusion detected.")
