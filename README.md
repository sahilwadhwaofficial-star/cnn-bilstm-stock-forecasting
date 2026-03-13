# 📈 CNN-BiLSTM Stock Forecasting

## 📌 Overview  
This project implements a hybrid **CNN–BiLSTM deep learning model** for stock price forecasting using OHLCV (Open, High, Low, Close, Volume) data.  
The model captures short-term patterns using CNN layers and long-term temporal dependencies using Bidirectional LSTM for accurate time-series forecasting.

---

## 🎯 Key Features  

• Hybrid CNN + BiLSTM architecture for time-series prediction  
• Uses OHLCV stock market data  
• 30-day future price forecasting  
• RMSE-based performance evaluation  
• Directional accuracy evaluation (Up/Down prediction)  
• Early stopping to prevent overfitting  

---

## 🛠 Tech Stack  

• Python  
• Pandas, NumPy  
• TensorFlow / Keras  
• Scikit-learn  
• Matplotlib  

---

## 📂 Project Structure  

cnn-bilstm-stock-forecasting/
│
├── data/
│   └── TATAMOTORS.NS.csv
│
├── src/
│   └── main.py
│
├── outputs/
│   ├── forecast_next_30_days.csv
│   └── (plots/screenshots)
│
├── notebooks/ (optional)
│   └── experimentation.ipynb
│
├── README.md
├── requirements.txt
└── LICENSE

---

## ⚙️ How to Run  

### 1️⃣ Clone Repository  
git clone https://github.com/sahilwadhwaofficial-star/cnn-bilstm-stock-forecasting  

### 2️⃣ Install Dependencies  
pip install -r requirements.txt  

### 3️⃣ Run Project  
python src/main.py  

---

## 📊 Results  

• RMSE: 74.53  
• Directional Accuracy: 55.93%  

The CNN–BiLSTM model demonstrates stable forecasting performance on unseen data.  
Directional accuracy exceeding the random baseline indicates the model effectively captures temporal market patterns.  
Early stopping was used to prevent overfitting and ensure robust generalization.

---

## 📌 Expected Output  

• RMSE printed in terminal  
• Directional accuracy printed  
• Forecast CSV saved in outputs/  
• Training loss plot displayed  

---

## 📄 License  
This project is licensed under the MIT License.
