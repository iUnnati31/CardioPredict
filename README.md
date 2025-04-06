# CardioPredict 

CardioPredict is a machine learning-based application designed to assess heart health and predict the risk of a heart attack. Built using Streamlit, the app provides a user-friendly interface for inputting health metrics and generating predictions based on trained machine learning models.

## 🟢 Features
- **Heart Health Risk Prediction**: Predicts the likelihood of a heart attack based on user input.
- **Detailed Feature Insights**: Provides explanations for each input parameter.
- **Interactive UI**: Modern, responsive interface with dark theme styling.

## 🛠️ Technologies Used
- Python
- Streamlit
- Pandas, Numpy
- Machine Learning (Pickle Model)

## 🚀 How to Run
```bash
# Clone the repository
git clone https://github.com/iUnnati31/CardioPredict.git

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

## 🐳 Run via Docker
```bash
# Pull the Docker image
docker pull unnatiag31/cardiopredict

# Run the container
docker run -p 8501:8501 unnatiag31/cardiopredict
```
Docker Hub Repository: CardioPredict on Docker Hub[https://hub.docker.com/repository/docker/unnatiag31/cardiopredict/general]

## 🎮 Usage
1. Input health metrics such as age, cholesterol, blood pressure, etc.
2. Click 'Analyze Heart Health Risk' to get a prediction.
3. View detailed insights in the sidebar.

## 📂 Project Structure
- **main.py**: Streamlit app for UI and prediction logic.
- **heart.pkl**: Trained machine learning model.


