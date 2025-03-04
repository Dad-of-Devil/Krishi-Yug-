from flask import Flask, request, render_template
import requests
import pandas as pd
import numpy as np
import pickle
import config2
import os

app = Flask(__name__)

# Load models safely
try:
    model_path = os.path.join(os.getcwd(), 'Machine learning model', 'model.pkl')
    scaler_path = os.path.join(os.getcwd(), 'Machine learning model', 'standardscaler.pkl')
    minmax_path = os.path.join(os.getcwd(), 'Machine learning model', 'minmaxscaler.pkl')

    print("Loading model...")
    model = pickle.load(open(model_path, 'rb'))
    sc = pickle.load(open(scaler_path, 'rb'))
    mx = pickle.load(open(minmax_path, 'rb'))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model, sc, mx = None, None, None  # Prevent crashes

# Fetch weather function
def weather_fetch(city_name):
    try:
        print(f"Fetching weather for city: {city_name}")
        api_key = config2.weather_api_key
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        response = requests.get(base_url, params={"q": city_name, "appid": api_key, "units": "metric"})
        data = response.json()

        if data.get("cod") != 200:
            print("City not found, using default weather values.")
            return 25.0, 50.0  # Default values

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        print(f"Weather: {temperature}°C, {humidity}% humidity")
        return temperature, humidity
    except Exception as e:
        print(f"Weather API error: {e}")
        return 25.0, 50.0  # Default values

# Home Page
@app.route('/')
def home():
    print("Rendering home page...")
    return render_template('new index.html', title='Krishi Yug - Home')

# Prediction Page
@app.route('/predict', methods=['POST'])
def prediction():
    print("Received prediction request...")
    if request.method == 'POST':
        try:
            N = int(request.form.get('nitrogen', 0))
            P = int(request.form.get('phosphorous', 0))
            K = int(request.form.get('potassium', 0))
            ph = float(request.form.get('ph', 0))
            rainfall = float(request.form.get('rainfall', 0))
            city = request.form.get("city", "")

            print(f"Inputs: N={N}, P={P}, K={K}, pH={ph}, Rainfall={rainfall}, City={city}")

            # Get weather data
            temperature, humidity = weather_fetch(city)

            # Prepare input
            feature_list = [N, P, K, temperature, humidity, ph, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)

            print("Transforming input...")
            mx_features = mx.transform(single_pred)
            sc_mx_features = sc.transform(mx_features)

            print("Making prediction...")
            prediction_result = model.predict(sc_mx_features)
            print(f"Prediction result: {prediction_result}")

            # Crop dictionary
            crop_dict = {
                1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
            }

            predicted_crop = crop_dict.get(prediction_result[0], "Unknown")
            result = f"{predicted_crop} is the best crop to be cultivated right there."

            print(f"Final Result: {result}")
            return render_template('new index.html', result=result)

        except Exception as e:
            print(f"Error in prediction: {e}")
            return f"Error in prediction: {e}", 500

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
