# app.py
# app.py
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
from PIL import Image

# Initialize models (cache to avoid reloading)
@st.cache_resource
def load_models():
    ner_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    ner_model = AutoModelForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    generator = pipeline("text2text-generation", 
                        model="google/flan-t5-base",
                        device=0 if torch.cuda.is_available() else -1)
    return ner_model, ner_tokenizer, generator

# --- Sidebar Section ---
with st.sidebar:
    try:
        logo_path = "Images/IridiumAILogo.png"
        iridium_logo = Image.open(logo_path)
        st.image(iridium_logo, use_container_width=True)  # Fixed parameter name
    except FileNotFoundError:
        st.warning("Logo image not found. Using text header instead.")
        st.title("IridiumAI Clinical Assistant")
    
    user_role = st.radio(
        "Select Your Role:",
        ("Clinician", "Pharmacist"),
        index=0,
        help="Tailor error detection to your professional needs"
    )

# --- Main UI Section ---
st.title(f"IridiumAI Clinical Notes Assistant - {user_role} Mode")
st.markdown("""
Detect potential errors in clinical notes using BioClinicalBERT and get correction suggestions with Flan-T5.
""")

# Define user_input BEFORE the button handler
user_input = st.text_area("Paste clinical note:", height=200)

# --- Helper Functions ---
def check_drug_interactions(text):
    interactions = []
    if 'warfarin' in text.lower() and ('ibuprofen' in text.lower() or 'naproxen' in text.lower()):
        interactions.append("Potential drug interaction: Warfarin with NSAIDs (risk of bleeding)")
    return interactions

def validate_diagnosis_codes(text):
    diagnosis_errors = []
    if 'ICD-10' not in text:
        diagnosis_errors.append("Missing ICD-10 diagnosis code")
    return diagnosis_errors

def detect_errors(text, model, tokenizer, user_role):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    
    entities = []
    for token, prediction in zip(inputs.tokens(), predictions[0].numpy()):
        if model.config.id2label[prediction] != "O":
            entities.append((token, model.config.id2label[prediction]))
    
    potential_errors = []
    for entity in entities:
        if "mg" in entity[0] and not any(char.isdigit() for char in entity[0]):
            potential_errors.append(f"Dosage format error: {entity[0]}")
        elif "QD" in entity[0]:
            potential_errors.append(f"Potentially ambiguous term: {entity[0]} (use 'daily' instead)")

    if user_role == "Pharmacist":
        potential_errors += check_drug_interactions(text)
    elif user_role == "Clinician":
        potential_errors += validate_diagnosis_codes(text)

    return potential_errors

def suggest_corrections(text, errors, generator):
    if not errors:
        return "No errors detected"
    
    prompt = f"Clinical note: {text}\nPotential issues: {', '.join(errors)}\nRewrite this note with corrections:"
    return generator(prompt, max_length=512)[0]['generated_text']

# --- Analysis Button Handler ---
if st.button("Analyze"):
    ner_model, ner_tokenizer, generator = load_models()
    
    with st.spinner("Analyzing..."):
        errors = detect_errors(user_input, ner_model, ner_tokenizer, user_role)
        
        if errors:
            corrected_text = suggest_corrections(user_input, errors, generator)
        else:
            corrected_text = "No significant errors detected"

    st.subheader("Analysis Results")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Potential Issues**")
        if errors:
            for error in errors:
                st.error(error)
        else:
            st.success("No critical issues found")
    
    with col2:
        st.markdown("**Suggested Corrections**")
        st.info(corrected_text)

st.markdown("---")
st.caption("Note: This is a prototype. Always verify with clinical experts.")