from flask import Flask,  render_template , request 
import requests
import pandas as pd
import numpy as np
import pickle
import config2


# Load machine learning models safely
import os

model_path = os.path.join(os.getcwd(), 'Machine learning model', 'model.pkl')
scaler_path = os.path.join(os.getcwd(), 'Machine learning model', 'standardscaler.pkl')
minmax_path = os.path.join(os.getcwd(), 'Machine learning model', 'minmaxscaler.pkl')

model = pickle.load(open(model_path, 'rb'))
sc = pickle.load(open(scaler_path, 'rb'))
mx = pickle.load(open(minmax_path, 'rb'))


# =========================================================================================
# Custom function to fetch weather data
def weather_fetch(city_name):
    try:
        api_key = config2.weather_api_key
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        response = requests.get(base_url, params={"q": city_name, "appid": api_key, "units": "metric"})
        data = response.json()

        if data.get("cod") != 200:
            return 25.0, 50.0  # Return default values if city not found

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        return temperature, humidity

    except Exception as e:
        print(f"Weather API error: {e}")
        return 25.0, 50.0  # Default values

# ===============================================================================================
# Initialize Flask app
app = Flask(__name__)

# Render home page
@app.route('/')
def home():
    title = 'Krishi Yug - Home'
    return render_template('new index.html', title=title)

# ===============================================================================================
# Render prediction page
@app.route('/predict', methods=['POST'])
def prediction():
    title = 'Krishi Yug - Crop Recommendation'

    if request.method == 'POST':
        try:
            N = int(request.form.get('nitrogen', 0))
            P = int(request.form.get('phosphorous', 0))
            K = int(request.form.get('potassium', 0))
            ph = float(request.form.get('ph', 0))
            rainfall = float(request.form.get('rainfall', 0))
            city = request.form.get("city", "")

            # Fetch weather data if city is provided
            if city:
                weather = weather_fetch(city)
                if weather:
                    temperature, humidity = weather
                else:
                    temperature, humidity = 25.0, 50.0  # Default values
            else:
                temperature, humidity = 25.0, 50.0  # Default values

            # Prepare input for prediction
            feature_list = [N, P, K, temperature, humidity, ph, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)

            mx_features = mx.transform(single_pred)
            sc_mx_features = sc.transform(mx_features)
            prediction_result = model.predict(sc_mx_features)

            # Crop dictionary
            crop_dict = {
                1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
            }

            # Generate prediction result
            predicted_crop = crop_dict.get(prediction_result[0], "Unknown")
            result = f"{predicted_crop} is the best crop to be cultivated right there."

            return render_template('new index.html', result=result)
        
        except Exception as e:
            return f"Error in prediction: {e}", 500

# ===============================================================================================
# Run the app
if __name__ == '__main__':
   app.run(debug=True, use_reloader=False)

