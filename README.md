🩺 AI Healthcare Chatbot

## 📸 Project Preview

| Home Screen | Diagnostic Report |
| :---: | :---: |
| ![Home Screen](image/home.png) | ![Report Screen](image/report.png) |
| *User Interface & Sidebar* | *AI-Generated Medical Analysis* |

Symptom-Based Disease Prediction using Random Forest
This project is an interactive healthcare solution that bridges the gap between Machine Learning and User-Centric Design. It uses a trained Random Forest Classifier to analyze user symptoms and provide a preliminary diagnostic report, complete with disease descriptions and actionable precautions.

🌟 Key Features
Intelligent Symptom Mapping: Uses difflib and keyword processing to map user-friendly symptom names to machine-readable data.

High-Accuracy Classifier: Employs a Random Forest Model with 300 estimators, trained on a comprehensive dataset of medical patterns.

Dynamic Data Loading: Loads medical descriptions and precautions directly from master CSV files to provide context-aware responses.

Interactive Chat UI: A clean Streamlit interface with a persistent sidebar for patient profiling and symptom selection.

Safety First: Includes automated disclaimers and "Model Confidence" scores to ensure users understand the tool is for demonstration purposes.

🛠️ Technical Stack
Frontend: Streamlit (Python)

Machine Learning: Scikit-learn (Random Forest Classifier, Label Encoding)

Data Processing: Pandas, NumPy

Natural Language: difflib for robust symptom matching

📊 Dataset & Model
The bot is powered by structured medical datasets:

Training.csv: Contains patterns for various diseases and their associated symptoms.

symptom_Description.csv: Provides formal definitions for conditions like Malaria, Diabetes, and Hypertension.

symptom_precaution.csv: Offers four levels of recommended actions for each predicted condition.

Model Logic: The model uses 300 estimators (trees) in a Random Forest ensemble to ensure stable and reliable predictions even with complex symptom overlaps.

📂 Repository Structure
streamlit_app.py: The main application file containing the UI logic and ML engine.

Data/: Contains Training.csv and Testing.csv used for model development.

MasterData/:

symptom_Description.csv: Detailed definitions of conditions.

symptom_precaution.csv: Specific steps to take for each diagnosis.

Symptom_severity.csv: Weights for different symptoms.

.gitignore: Configured to exclude heavy virtual environments (venv/) and local caches.

💻 Installation & Setup
1. Clone the Repository
Bash
git clone https://github.com/AyushKhaitan1/AI-Mental-healthcare-Chatbot.git
cd AI-Mental-healthcare-Chatbot
2. Set Up Virtual Environment
Bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Run the Application
Bash
streamlit run streamlit_app.py

⚠️ Medical Disclaimer
This AI tool is for educational and portfolio purposes only. The results provided are based on statistical patterns and should not be considered professional medical advice. In case of an emergency, please contact your local healthcare provider immediately.

