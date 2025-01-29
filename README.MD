
![image](https://github.com/ahmadalis2016/Iridium-AI-Image-Analysis/assets/130319416/3590b637-b72a-4a41-86ba-a556e9c22016) 

## Iridium AI: Clinical Notes Assistant 🩺🤖

An AI-powered application for detecting and correcting errors in clinical notes using state-of-the-art language models.

![Demo Screenshot](Images/screenshot1.png) *(Add screenshot after testing)*

## Features
- **Role-Based Error Detection**: Tailored checks for:
  - **Clinicians**: Diagnosis code validation, lab result formatting
  - **Pharmacists**: Drug interaction checks, dosage validation
- **Error Types Detected**:
  - Missing/invalid dosage information
  - Ambiguous medical abbreviations (e.g., QD, TID)
  - Drug-drug interactions
  - Missing ICD-10 codes
  - Incomplete lab values
- **AI-Powered Corrections**: FLAN-T5 generated suggestions
- **Interactive UI**: Streamlit-based web interface
- **Expandable Architecture**: Easy to add new validation rules


## Usage

Interface Guide:
Left Sidebar:
 - Select your professional role
 - View application info

Main Panel:

 - Paste clinical notes in text area
 - Click "Analyze" to run checks
 - View errors and AI suggestions side-by-side

Example Input:
Patient prescribed warfarin 5mg daily and ibuprofen 400mg TID. 
Recent INR 2.3. Allergy: sulfa drugs. Follow-up in 14 days.

## Contributing
Ahmad A Shaik, Ph. D.

## Disclaimer ⚠️
This is a prototype system for demonstration purposes only. Not for clinical use. Always verify AI suggestions with:

 - Institutional guidelines
 - Up-to-date drug databases
 - Clinical judgment
 
## License
This project is licensed under the MIT License. 
