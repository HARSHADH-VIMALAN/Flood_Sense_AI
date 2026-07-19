# 🌊 FloodSense AI - Urban Flood Risk Prediction System

### AI-Powered Urban Flood Risk Prediction & Disaster Assistance System

FloodSense AI is an intelligent flood prediction and emergency assistance platform built using **Python, Streamlit, Scikit-learn, Plotly, and Groq AI**. It predicts urban flood risks using environmental and geographical data while also providing real-time AI-powered disaster guidance, safety recommendations, and emergency support.

---

## 📌 Features

### 🌧️ Flood Risk Prediction
- Predicts flood occurrence using Machine Learning.
- Supports Random Forest and Gradient Boosting algorithms.
- Displays flood probability and risk levels.
- Generates easy-to-understand prediction explanations.

### 📊 Interactive Dashboard
- Dataset overview
- Flood vs Safe distribution
- Rainfall trend analysis
- Flood statistics
- Key insights

### 🗺️ Interactive Flood Map
- Plot flood locations using GPS coordinates
- Flood hotspot heatmap
- High-risk area identification

### 📈 Risk Factor Analysis
- Feature importance visualization
- Correlation analysis
- Top flood contributing factors

### 🤖 AI Performance Evaluation
- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- Confusion Matrix
- Cross Validation Score

### 📂 Data Explorer
- Filter flood and safe events
- Sort datasets
- Download filtered CSV/TSV
- Data quality analysis

### 🤖 Flood AI Assistant (Groq AI)
- Flood safety guidance
- Rescue assistance
- Emergency kit generation
- Flood first aid
- Nearby hospitals
- Police stations
- Fire stations
- Relief camps
- Government support information
- Children safety
- Pet safety
- Vehicle safety

---

# 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- OpenPyXL
- Groq AI (Llama 3.3 70B)
- HTML & CSS

---

# 📂 Dataset Requirements

The uploaded Excel dataset should contain the following columns:

| Column |
|---------|
| Latitude |
| Longitude |
| Rainfall (mm) |
| Temperature (°C) |
| Humidity (%) |
| River Discharge (m³/s) |
| Water Level (m) |
| Elevation (m) |
| Land Cover |
| Soil Type |
| Population Density |
| Infrastructure |
| Historical Floods |
| Flood Occurred |

---

# ⚙️ Machine Learning Pipeline

1. Upload Dataset
2. Data Cleaning
3. Label Encoding
4. Feature Scaling
5. Train/Test Split
6. Model Training
7. Flood Prediction
8. Performance Evaluation
9. Risk Visualization
10. AI Disaster Assistance

---

# 📊 Evaluation Metrics

- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- AUC Score
- Cross Validation Score

---

# 🤖 AI Assistant Capabilities

The integrated Groq AI assistant can answer questions related to:

- Flood preparedness
- Rescue guidance
- Emergency response
- Evacuation planning
- Nearby emergency centres
- Government helplines
- Weather precautions
- Flood first aid
- Children safety
- Pet safety
- Vehicle protection
- Emergency kits

---

# 📸 Application Modules

- 🏠 Dashboard
- 🔍 Predict Risk
- 🗺️ Flood Map
- 📊 Risk Factors
- 🤖 AI Performance
- 📋 Data Explorer
- 🤖 AI Assistant

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/FloodSense-AI.git
```

Move into the project folder

```bash
cd FloodSense-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 🔑 API Configuration

Replace the following line inside `app.py`

```python
GROQ_API_KEY = "YOUR_API_KEY"
```

with your own Groq API Key.

---

# 📁 Project Structure

```
FloodSense-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── dataset.xlsx
└── assets/
```

---

# 🌍 Future Enhancements

- Live weather API integration
- Google Maps integration
- Real-time flood alerts
- SMS notification system
- Satellite imagery analysis
- IoT sensor integration
- Mobile application
- Multi-language AI assistant

---

# 🎯 Applications

- Disaster Management Authorities
- Municipal Corporations
- Smart Cities
- Environmental Monitoring
- Urban Planning
- Climate Risk Assessment
- Emergency Response Teams
- Research Institutions

---

# 👨‍💻 Author

**Harshadh Vimalan**

B.Sc Artificial Intelligence & Machine Learning

Final Year Project

---

# 📄 License

This project is developed for educational and research purposes.
