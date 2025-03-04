# Krishi-Yug-
# 🌾 Crop Recommendation System

## 📌 Overview  
The **Crop Recommendation System** is a machine learning-based web application that suggests the most suitable crop for farming based on environmental factors such as soil type, temperature, humidity, rainfall, and pH levels. It helps farmers and agricultural researchers make data-driven decisions to optimize crop yield.

## 🚀 Features  
- 🌱 Predicts the best crop based on soil and climate conditions.  
- 📊 Utilizes machine learning models for accurate recommendations.  
- 🖥️ Web-based interface using Flask.  
- 📉 Data preprocessing and exploratory data analysis (EDA).  
- 💾 Uses a dataset with soil and climatic conditions.  

## ⚙️ Technologies Used  
- **Python** 🐍  
- **Flask** 🌐 (for web application)  
- **Scikit-learn** 🤖 (for machine learning model)  
- **Pandas & NumPy** 📊 (for data handling)  
- **Matplotlib & Seaborn** 📈 (for data visualization)  
- **HTML, CSS, JavaScript** 🎨 (for UI/UX)  

## 📂 Project Structure  
📂 Crop-Recommendation-System
├── 📁 static # CSS, JavaScript, images
├── 📁 templates # HTML templates
├── 📁 models # Machine learning model
├── 📄 app.py # Flask application
├── 📄 requirements.txt # Dependencies
├── 📄 README.md # Project documentation
├── 📄 dataset.csv # Crop dataset



## 🛠️ Installation & Setup  
### Prerequisites  
Ensure you have **Python 3.x** installed.  

### Steps to Set Up Locally  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/Crop-Recommendation-System.git
   cd Crop-Recommendation-System
2 . install the dependencies
   pip install -r requirements.txt

3. Run the Flask application:
   python app.py
   
4. open in :   http://127.0.0.1:5000




📌 How It Works
User Input: Enter values like temperature, humidity, rainfall, and soil pH.
Prediction: The ML model predicts the best crop for the given inputs.
Results Display: The recommended crop is shown with additional insights.
📊 Dataset
The model is trained on an agricultural dataset that includes various parameters such as Nitrogen, Phosphorus, Potassium (NPK levels), Temperature, Humidity, pH, and Rainfall.

🤝 Contributing
Contributions are welcome! Follow these steps:

Fork the repository 🍴
Create a new branch: git checkout -b feature-name
Commit your changes: git commit -m "Add feature"
Push to your branch: git push origin feature-name
Open a Pull Request 🚀
