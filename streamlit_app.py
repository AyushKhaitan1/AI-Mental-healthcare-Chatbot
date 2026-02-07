import streamlit as st
import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
import warnings
from nltk.corpus import wordnet

# --- Suppress Scikit-learn UserWarning and DeprecationWarning ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore", 
    message="X does not have valid feature names, but RandomForestClassifier was fitted with feature names"
)

# --- CONFIGURATION ---
DATA_PATH = 'Data/Training.csv'
DESCRIPTION_PATH = 'MasterData/symptom_Description.csv'
PRECAUTION_PATH = 'MasterData/symptom_precaution.csv'

# --- Custom CSS for Normalized Styling ---
def apply_custom_css():
    st.markdown("""
    <style>
    /* ------------------- GENERAL APP STYLING ------------------- */
    
    /* Remove background and sidebar color overrides to use Streamlit's default theme */
    
    /* Main Title Color: Use a neutral dark color, but keep bold effect */
    .st-emotion-cache-10trblm {
        color: #262730; /* Streamlit default text color */
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.05);
    }
     /* ------------------- CHAT BUBBLE STYLING ------------------- */
    
    /* Assistant (Bot) Bubble: ENHANCED VISIBILITY */
    .stChatMessage [data-testid="stMarkdownContainer"] {
        background-color: #e6f9ff; /* Very Pale Blue for contrast */
        color: #004085; /* Strong dark blue text for readability */
        border-left: 4px solid #1a9cff; /* Streamlit's primary blue accent */
        border-radius: 8px;
        padding: 10px;
    }
    
    /* User Bubble: Light neutral background (White) */
    .st-emotion-cache-1v0u0j4 {
        background-color: #ffffff; /* White */
        color: #262730; /* Dark text */
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #ced4da;
    }
    
    /* Ensure the text inside the chat input box is dark */
    [data-testid="stTextInput"] > div > div > input {
        color: #262730 !important; 
    }
    /* Warning/Precautions Style */
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. CORE CACHING FUNCTIONS (Load Data/Model Once) ---

@st.cache_resource
def load_and_train_model():
    """Loads data, trains the Random Forest model, and prepares dictionaries."""
    
    training = pd.read_csv(DATA_PATH)
    training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
    training = training.loc[:, ~training.columns.duplicated()]

    cols = training.columns[:-1]
    x = training[cols]
    y = training['prognosis']

    le = preprocessing.LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    x_train, _, y_train, _ = train_test_split(x, y_encoded, test_size=0.33, random_state=42)
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(x_train, y_train)

    symptoms_dict = {symptom: idx for idx, symptom in enumerate(cols)}
    
    user_friendly_symptoms = [s.replace('_', ' ').title() for s in cols]
    
    return model, le, cols, symptoms_dict, user_friendly_symptoms

@st.cache_data
def load_master_data():
    """Loads precaution and description files."""
    description_dict = {}
    precaution_dict = {}
    
    # Load Description Data
    with open(DESCRIPTION_PATH, mode='r') as file:
        reader = csv.reader(file)
        next(reader) 
        for row in reader:
            if row:
                disease = row[0].strip()
                description_dict[disease] = row[1].strip() if len(row) > 1 else "No description available."
    
    # Load Precaution Data
    with open(PRECAUTION_PATH, mode='r') as file:
        reader = csv.reader(file)
        next(reader) 
        for row in reader:
            if row:
                disease = row[0].strip()
                precaution_dict[disease] = [p.strip() for p in row[1:] if p.strip()]

    return description_dict, precaution_dict

# --- 2. HELPER FUNCTIONS ---

def get_machine_symptom(user_symptom, symptoms_dict):
    """Maps a user-friendly symptom string to a machine-readable keyword."""
    machine_key = user_symptom.lower().replace(' ', '_').strip()
    
    if machine_key in symptoms_dict:
        return machine_key
    
    return None

def check_pattern(selected_symptoms, model, le, all_symptom_cols, symptoms_dict):
    """
    Takes a list of selected user symptoms and runs the prediction logic.
    
    Returns: prediction_index, matched_symptoms_list (user-friendly format), confidence
    """
    
    input_vector = np.zeros(len(all_symptom_cols))
    matched_symptoms_list = []
    
    for user_symptom in selected_symptoms:
        machine_symptom = get_machine_symptom(user_symptom, symptoms_dict)
        
        if machine_symptom and machine_symptom in symptoms_dict:
            input_vector[symptoms_dict[machine_symptom]] = 1
            matched_symptoms_list.append(user_symptom)
        else:
            # Although using multiselect reduces need for fuzzy matching, this ensures robustness
            close_matches = get_close_matches(user_symptom.lower().replace(' ', '_'), symptoms_dict.keys(), cutoff=0.8)
            if close_matches:
                best_match = close_matches[0]
                input_vector[symptoms_dict[best_match]] = 1
                matched_symptoms_list.append(user_symptom)


    if sum(input_vector) == 0:
        return None, matched_symptoms_list, 0.0 

    input_data = input_vector.reshape(1, -1)
    
    # Predict the disease index and get probability array
    prediction_array = model.predict_proba(input_data)[0]
    prediction_index = np.argmax(prediction_array)
    confidence = prediction_array[prediction_index]
    
    return prediction_index, matched_symptoms_list, confidence

# --- 3. STREAMLIT CHAT UI LAYOUT ---

# Apply custom styling
apply_custom_css()

# Initialize state for conversation history and user profile
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_profile" not in st.session_state:
    # Set default age to 1 to avoid StreamlitValueBelowMinError
    st.session_state.user_profile = {"name": "", "age": 1, "gender": "N/A"} 

# Load model and data
model, le, all_symptom_cols, symptoms_dict, user_friendly_symptoms = load_and_train_model()
description_dict, precaution_dict = load_master_data()


st.set_page_config(
    page_title="AI Health Chatbot", 
    layout="wide",
    initial_sidebar_state="expanded" # Keep sidebar open by default
)

# --- Sidebar for Interactivity ---
with st.sidebar:
    st.title("Patient Input 🧑‍⚕️")
    st.markdown("---")
    
    # Input Form 
    with st.form("Profile_Form"):
        st.subheader("1. Personal Details")
        st.session_state.user_profile["name"] = st.text_input("Name:", value=st.session_state.user_profile["name"])
        
        # Age input with min_value=1
        st.session_state.user_profile["age"] = st.number_input("Age:", min_value=1, max_value=120, value=st.session_state.user_profile["age"])
        
        st.session_state.user_profile["gender"] = st.selectbox("Gender:", ["Male", "Female", "Other", "N/A"], index=["Male", "Female", "Other", "N/A"].index(st.session_state.user_profile["gender"]))
        
        st.markdown("---")
        st.subheader("2. Select Symptoms")
        
        # Multiselect for interactive symptom input
        selected_symptoms = st.multiselect(
            "Select ALL symptoms you are experiencing:",
            user_friendly_symptoms,
            default=st.session_state.get('selected_symptoms', []) # Persist selection
        )
        st.session_state.selected_symptoms = selected_symptoms # Save selection state
        
        diagnosis_button = st.form_submit_button("🩺 Diagnose Condition", type="primary")

    st.markdown("---")
    if st.button("Start New Consultation 🧼"):
        st.session_state.messages = []
        st.session_state.selected_symptoms = []
        st.session_state.user_profile = {"name": "", "age": 1, "gender": "N/A"}
        st.rerun()

# --- Main Chat Interface ---
st.title("🩺 AI Healthcare Chatbot 🩺")
st.caption("Using Random Forest Classifier for Symptom Analysis")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Initial Message ---
if not st.session_state.messages:
    initial_message = (
        "👋 Welcome! I am Dr. A.I., your virtual symptom checker. "
        "Please enter your **Name, Age, and Gender** in the sidebar, and then **select all your symptoms** from the dropdown menu. "
        "Once done, click the **'Diagnose Condition'** button!"
    )
    st.session_state.messages.append({"role": "assistant", "content": initial_message})
    with st.chat_message("assistant"):
        st.markdown(initial_message)

# --- Diagnosis Logic (Triggered by Sidebar Button) ---
if diagnosis_button:
    # 1. Validate Input
    if not selected_symptoms:
        st.error("🚨 Please select at least one symptom to run the diagnosis.")
    else:
        # 2. Add user message to chat history
        user_summary = (
            f"**Patient Name:** {st.session_state.user_profile['name']} | **Age:** {st.session_state.user_profile['age']} | **Gender:** {st.session_state.user_profile['gender']}\n\n"
            f"**Symptoms Reported:** {', '.join(selected_symptoms)}"
        )
        st.session_state.messages.append({"role": "user", "content": user_summary})
        with st.chat_message("user"):
            st.markdown(user_summary)
            
        # 3. Run Prediction Logic
        with st.spinner('Analyzing patterns and compiling diagnostic report...'):
            prediction_index, matched_symptoms_list, confidence = check_pattern(
                selected_symptoms, model, le, all_symptom_cols, symptoms_dict
            )

        # 4. Generate Assistant Response
        with st.chat_message("assistant"):
            if prediction_index is None:
                response = "❌ **Diagnosis Failed:** The symptoms you provided did not trigger any known pattern in our model. Please ensure your selected symptoms are accurate."
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)
            else:
                # Handle Successful Prediction
                predicted_disease = le.inverse_transform([prediction_index])[0].strip()
                description = description_dict.get(predicted_disease, "Description not found.")
                precautions = precaution_dict.get(predicted_disease, ["No specific precautions found."])
                
                confidence_percent = f"{confidence * 100:.2f}%"
                
                # --- Format Output (More Interesting Answer) ---
                response_content = f"""
                ## ✅ AI Diagnostic Report (Confidence: {confidence_percent})

                Based on the analysis of your symptoms, the primary predicted condition is:
                ### **{predicted_disease.title()}**

                ---
                #### 📝 About the Condition
                * {description}

                #### 🛡️ Precautions & Next Steps
                """
                for p in precautions:
                     response_content += f"\n* **{p.title()}**"
                
                response_content += f"\n\n---"
                response_content += f"\n\n🩺 **Model Confidence:** Our Random Forest model predicts this with **{confidence_percent} confidence**. Remember: this is a demonstration tool. For a definitive diagnosis, please consult a medical doctor immediately."
                
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                st.markdown(response_content)