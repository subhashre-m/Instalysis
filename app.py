import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import asyncio
import logging
import os
import concurrent.futures
import json
from collections import Counter
import re

import torch
from transformers import AutoTokenizer
import google.generativeai as genai
from safetensors.torch import load_file
from weasyprint import HTML  

from instaloader_fetcher import fetch_captions
from preprocess import preprocess_captions
from fusion_model import FusionClassifier, FusionConfig

###############################################################################
# Environment and Logging Setup
###############################################################################
GENAI_API_KEY = "AIzaSyAQDazslbfHxZ3OHKYlbU00iJrqb_YotYw"  # Replace with your Gemini API key
genai.configure(api_key=GENAI_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Instalysis", layout="wide")

# Initialize session state variables if they don't exist
for key in ["messages", "ig_analysis_done", "user_mood", "username", "auth_type", "recommendations", "report_html", "current_page"]:
    if key not in st.session_state:
        if key == "messages":
            st.session_state[key] = []
        elif key == "current_page":
            # Default to "Analysis" if no analysis has been done yet; else "Chat"
            st.session_state[key] = "Analysis"
        else:
            st.session_state[key] = ""

###############################################################################
# Minimal Styling (Only the report is fancy)
###############################################################################
st.markdown("""
    <style>
    /* Minimal styling for the main UI */
    .report-box {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        border: 2px solid #34568b;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        font-family: 'Courier New', Courier, monospace;
    }
    </style>
""", unsafe_allow_html=True)

###############################################################################
# Load Fusion Model and Tokenizers
###############################################################################
model_path = "./final_fusion"  # Adjust if needed

config = FusionConfig(
    num_labels=5,
    xlmr_checkpoint="xlm-roberta-large",
    mbert_checkpoint="bert-base-multilingual-cased",
    xlmr_hidden_size=1024,
    mbert_hidden_size=768,
    label2id={
        "Normal": 0,
        "Happy": 1,
        "Stressed": 2,
        "Moderately Depressed": 3,
        "Severely Depressed": 4
    },
    id2label={
        0: "Normal",
        1: "Happy",
        2: "Stressed",
        3: "Moderately Depressed",
        4: "Severely Depressed"
    }
)

model = FusionClassifier(config)
state_dict = load_file(os.path.join(model_path, "model.safetensors"))
model.load_state_dict(state_dict)
model.eval()

tokenizer_xlmr = AutoTokenizer.from_pretrained("xlm-roberta-large")
tokenizer_mbert = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
if tokenizer_mbert.pad_token is None:
    tokenizer_mbert.pad_token = tokenizer_mbert.eos_token

label_map = {
    0: "Normal",
    1: "Happy",
    2: "Stressed",
    3: "Moderately Depressed",
    4: "Severely Depressed"
}

###############################################################################
# Prediction Function: Determine Mental Health Condition from Captions
###############################################################################
def predict_state_of_mind(captions):
    # Preprocess all captions in batch
    cleaned_captions = [preprocess_captions([caption])[0] for caption in captions]
    
    # Tokenize in batch for both tokenizers
    xlmr_inputs = tokenizer_xlmr(
        cleaned_captions,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=128
    )
    mbert_inputs = tokenizer_mbert(
        cleaned_captions,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=128
    )
    
    # Move model and tensors to GPU if available
    if torch.cuda.is_available():
        xlmr_inputs = {key: val.to('cuda') for key, val in xlmr_inputs.items()}
        mbert_inputs = {key: val.to('cuda') for key, val in mbert_inputs.items()}
        model.to('cuda')
    else:
        model.to('cpu')
    
    with torch.no_grad():
        outputs = model(
            xlmr_input_ids=xlmr_inputs["input_ids"],
            xlmr_attention_mask=xlmr_inputs["attention_mask"],
            mbert_input_ids=mbert_inputs["input_ids"],
            mbert_attention_mask=mbert_inputs["attention_mask"]
        )
    
    logits = outputs["logits"]
    hard_preds = torch.argmax(logits, dim=-1).cpu().numpy()
    vote_counts = Counter(hard_preds)
    majority_label, count = vote_counts.most_common(1)[0]
    if count >= len(hard_preds) / 2:
        final_label = majority_label
    else:
        avg_logits = torch.mean(logits, dim=0)
        final_label = torch.argmax(avg_logits).item()
    
    return label_map[final_label]

###############################################################################
# Get Recommendations Function (from Code1)
###############################################################################
def get_recommendations(state):
    try:
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"Provide 3 concise recommendations for someone feeling '{state}' to improve mental well-being. "
            "Be direct and supportive."
        )
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

###############################################################################
# Caching Gemini responses to reduce redundant calls
###############################################################################
@st.cache_data(show_spinner=False)
def cached_gemini_generate(prompt, model_name="gemini-1.5-flash"):
    try:
        model_gemini = genai.GenerativeModel(model_name)
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

###############################################################################
# Gemini Chatbot Logic with Refined Empathetic Prompt
###############################################################################
def gemini_chatbot_logic(chat_log, user_input):
    check_prompt = (
        "Determine if the message is strictly about: "  
        "1. Mental health (e.g., anxiety, depression, trauma), "  
        "2. Emotional struggles (e.g., grief, loneliness, stress), "  
        "3. Medical concerns (e.g., symptoms, diagnoses, treatment clarification), "  
        "4. Crisis support (e.g., suicidal thoughts, self-harm urges), "  
        "5. Seeking help/therapy advice, or "  
        "6. A simple greeting (e.g., 'hi', 'hello'). "  
        "Ignore non-medical/non-emotional topics (e.g., tech, finance, casual chat). "  
        "Answer only 'Yes' or 'No':\n"  
        f"Message: '{user_input}'"
    )
    check_response = cached_gemini_generate(check_prompt)
    if check_response.lower().startswith("no"):
        return "Please ask mental health related questions or greet me with hi/hello."
    
    refined_prompt = (
        "You are a warm, empathetic mental health counsellor with specialized training in active listening and crisis intervention. "
        "Respond with GENUINE CARE using this framework:\n\n"
        "- Start by acknowledging the user's feelings (Emotional Mirroring).\n"
        "2. **Evidence-Based Technique** (choose ONE): Grounding, Reframing, or Containment.\n"
        "- Offer a small, supportive action or resource (Collaborative Next Steps).\n\n"
        "4. **Crisis Protocol**: If needed, provide immediate safety advice.\n\n"
        "Do NOT include the labels (Emotional Mirroring), (Evidence-Based Technique), (Collaborative Next Steps), or (Crisis Protocol) in your response. Weave these elements naturally into the conversation."
        "Current User Message:\n"
        f"'{user_input}'\n\n"
        "Naturally weave these elements into a caring and supportive response."
    )
    
    # Format chat history for the prompt
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_log])
    
    # Restructure prompt to better incorporate conversation history
    full_prompt = f"""
{refined_prompt}

Consider the entire conversation history below when formulating your response:

{history_text}
User: {user_input}
Therapist:"""
    
    response = cached_gemini_generate(full_prompt)
    return response

###############################################################################
# Generate Comprehensive Mental State JSON Report
###############################################################################
def get_comprehensive_report(chat_log, username):
    transcript = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_log])
    prompt = (
        "You are an expert mental health analyst. Based on the following conversation transcript, "
        "generate a detailed report in JSON format with the following structure:\n\n"
        "{\n"
        "  \"report\": {\n"
        "    \"metadata\": {\n"
        "      \"report_generated_on\": \"<ISO_TIMESTAMP>\",\n"
        "      \"session_id\": \"<SESSION_ID>\",\n"
        "      \"conversation_duration\": \"<DURATION>\",\n"
        "      \"assessment_tool\": \"Instalysis: AI Counsellor\",\n"
        "      \"version\": \"1.0\"\n"
        "    },\n"
        "    \"patient_summary\": {\n"
        "      \"patient_name\": \"<USERNAME>\",\n"
        "      \"mental_state_overview\": \"<OVERVIEW>\"\n"
        "    },\n"
        "    \"clinical_impression\": {\n"
        "      \"diagnostic_summary\": \"<SUMMARY>\",\n"
        "      \"emotional_trends\": {\n"
        "        \"predominant_emotions\": [\"<EMOTION1>\", \"<EMOTION2>\", \"<EMOTION3>\"],\n"
        "        \"mood_variability\": \"<DESCRIPTION>\"\n"
        "      }\n"
        "    },\n"
        "    \"conversation_analysis\": {\n"
        "      \"summary\": \"<ANALYSIS>\",\n"
        "      \"key_emotional_cues\": [\n"
        "        {\n"
        "          \"timestamp\": \"<TIMESTAMP>\",\n"
        "          \"utterance\": \"<UTTERANCE>\",\n"
        "          \"clinical_note\": \"<NOTE>\"\n"
        "        }\n"
        "      ]\n"
        "    },\n"
        "    \"risk_assessment\": {\n"
        "      \"suicide_risk\": \"<RISK>\",\n"
        "      \"self_harm_risk\": \"<RISK>\",\n"
        "      \"other_risks\": \"<RISK_DETAILS>\"\n"
        "    },\n"
        "    \"strengths_and_resources\": {\n"
        "      \"strengths\": \"<STRENGTHS>\",\n"
        "      \"support_system\": \"<SUPPORT_DETAILS>\"\n"
        "    },\n"
        "    \"treatment_recommendations\": {\n"
        "      \"therapy_recommendations\": [\"<THERAPY1>\", \"<THERAPY2>\"],\n"
        "      \"medication_consideration\": \"<MEDICATION>\",\n"
        "      \"lifestyle_recommendations\": [\"<LIFESTYLE1>\", \"<LIFESTYLE2>\", \"<LIFESTYLE3>\"]\n"
        "    },\n"
        "    \"additional_resources\": [\n"
        "      {\n"
        "        \"name\": \"<RESOURCE_NAME>\",\n"
        "        \"contact\": \"<CONTACT>\",\n"
        "        \"website\": \"<WEBSITE>\"\n"
        "      }\n"
        "    ],\n"
        "    \"conclusion\": \"<CONCLUSION>\",\n"
        "    \"disclaimer\": \"This report is generated by an AI system and is for informational purposes only. It is not a substitute for professional medical advice.\"\n"
        "  }\n"
        "}\n\n"
        "Use the following transcript:\n"
        f"{transcript}\n\n"
        "Replace placeholders with your analysis. Ensure the patient_name is set to the username: {username}."
    )
    response = cached_gemini_generate(prompt)
    return response

###############################################################################
# Helper: Format Report as a Beautiful Medical Report (Professional Style)
###############################################################################
def format_report_as_html(report):
    # report is expected to be a dictionary (the value from report_json["report"])
    meta = report.get("metadata", {})
    patient = report.get("patient_summary", {})
    clinical = report.get("clinical_impression", {})
    conversation = report.get("conversation_analysis", {})
    risk = report.get("risk_assessment", {})
    strengths = report.get("strengths_and_resources", {})
    treatment = report.get("treatment_recommendations", {})
    resources = report.get("additional_resources", [])
    conclusion = report.get("conclusion", "")
    disclaimer = report.get("disclaimer", "")

    html_content = f"""
    <html>
    <head>
      <meta charset="UTF-8" />
      <style>
        @page {{
            margin: 2.5cm;
            size: A4;
        }}
        body {{
            font-family: 'Times New Roman', Times, serif;
            margin: 0;
            padding: 0;
            color: #000000;
            background-color: #ffffff;
            line-height: 1.6;
            font-size: 12pt;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid #000000;
        }}
        .header h1 {{
            font-size: 32pt;
            margin: 0;
            padding: 20px 0 10px 0;
            color: #000000;
            font-weight: bold;
            letter-spacing: 2px;
        }}
        .header h2 {{
            font-size: 24pt;
            margin: 0;
            padding: 10px 0;
            color: #000000;
            font-weight: bold;
        }}
        .metadata {{
            font-size: 10pt;
            color: #666666;
            margin: 20px 0;
            text-align: center;
        }}
        .section {{
            margin: 30px 0;
            page-break-inside: avoid;
        }}
        .section-title {{
            font-size: 16pt;
            font-weight: bold;
            color: #000000;
            margin: 25px 0 15px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid #000000;
        }}
        .content-block {{
            margin: 15px 0;
            padding-left: 20px;
        }}
        .content-block p {{
            margin: 10px 0;
            text-align: justify;
        }}
        .bold {{
            font-weight: bold;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #000000;
            font-size: 10pt;
            text-align: center;
            color: #000000;
        }}
        ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        li {{
            margin: 8px 0;
            text-align: justify;
        }}
        .disclaimer {{
            font-size: 10pt;
            font-style: italic;
            margin-top: 30px;
            padding: 15px;
            border: 1px solid #000000;
            background-color: #f9f9f9;
            text-align: justify;
        }}
        .risk-assessment {{
            background-color: #f8f8f8;
            padding: 15px;
            border-left: 4px solid #000000;
            margin: 15px 0;
        }}
        .emotional-cue {{
            margin: 15px 0;
            padding: 10px;
            border-left: 3px solid #666666;
            background-color: #fafafa;
        }}
        .resource-item {{
            margin: 10px 0;
            padding: 10px;
            border: 1px solid #dddddd;
            background-color: #ffffff;
        }}
      </style>
    </head>
    <body>

      <!-- Header -->
      <div class="header">
        <h1>INSTALYSIS</h1>
        <h2>Mental Health Assessment Report</h2>
        <div class="metadata">
          <p>Report Generated: {meta.get("report_generated_on", "N/A")} | Session ID: {meta.get("session_id", "N/A")}</p>
        </div>
      </div>

      <!-- Patient Information -->
      <div class="section">
        <div class="section-title">Patient Information</div>
        <div class="content-block">
          <p><span class="bold">Patient Name:</span> {patient.get("patient_name", "N/A")}</p>
          <p><span class="bold">Mental State Overview:</span></p>
          <p>{patient.get("mental_state_overview", "N/A")}</p>
        </div>
      </div>

      <!-- Clinical Impression -->
      <div class="section">
        <div class="section-title">Clinical Impression</div>
        <div class="content-block">
          <p><span class="bold">Diagnostic Summary:</span></p>
          <p>{clinical.get("diagnostic_summary", "N/A")}</p>
          <p><span class="bold">Emotional Assessment:</span></p>
          <ul>
            <li><span class="bold">Predominant Emotions:</span> {", ".join(clinical.get("emotional_trends", {}).get("predominant_emotions", []))}</li>
            <li><span class="bold">Mood Variability:</span> {clinical.get("emotional_trends", {}).get("mood_variability", "N/A")}</li>
          </ul>
        </div>
      </div>

      <!-- Conversation Analysis -->
      <div class="section">
        <div class="section-title">Conversation Analysis</div>
        <div class="content-block">
          <p><span class="bold">Summary:</span></p>
          <p>{conversation.get("summary", "N/A")}</p>
          <p><span class="bold">Key Emotional Cues:</span></p>
          <div class="emotional-cues">
    """
    for cue in conversation.get("key_emotional_cues", []):
        timestamp = cue.get("timestamp", "")
        utterance = cue.get("utterance", "")
        note = cue.get("clinical_note", "")
        html_content += f"""
            <div class="emotional-cue">
              <p><span class="bold">Time {timestamp}:</span></p>
              <p><span class="bold">Statement:</span> {utterance}</p>
              <p><span class="bold">Clinical Note:</span> {note}</p>
            </div>"""
    
    html_content += f"""
          </div>
        </div>
      </div>

      <!-- Risk Assessment -->
      <div class="section">
        <div class="section-title">Risk Assessment</div>
        <div class="content-block">
          <div class="risk-assessment">
            <ul>
              <li><span class="bold">Suicide Risk:</span> {risk.get("suicide_risk", "N/A")}</li>
              <li><span class="bold">Self-Harm Risk:</span> {risk.get("self_harm_risk", "N/A")}</li>
              <li><span class="bold">Other Risks:</span> {risk.get("other_risks", "N/A")}</li>
            </ul>
          </div>
        </div>
      </div>

      <!-- Strengths and Resources -->
      <div class="section">
        <div class="section-title">Strengths and Resources</div>
        <div class="content-block">
          <p><span class="bold">Identified Strengths:</span></p>
          <p>{strengths.get("strengths", "N/A")}</p>
          <p><span class="bold">Support System:</span></p>
          <p>{strengths.get("support_system", "N/A")}</p>
        </div>
      </div>

      <!-- Treatment Recommendations -->
      <div class="section">
        <div class="section-title">Treatment Recommendations</div>
        <div class="content-block">
          <p><span class="bold">Therapy Recommendations:</span></p>
          <ul>
    """
    for therapy in treatment.get("therapy_recommendations", []):
        html_content += f"<li>{therapy}</li>"
    
    html_content += f"""
          </ul>
          <p><span class="bold">Medication Consideration:</span></p>
          <p>{treatment.get("medication_consideration", "N/A")}</p>
          <p><span class="bold">Lifestyle Recommendations:</span></p>
          <ul>
    """
    for lifestyle in treatment.get("lifestyle_recommendations", []):
        html_content += f"<li>{lifestyle}</li>"
    
    html_content += f"""
          </ul>
        </div>
      </div>

      <!-- Additional Resources -->
      <div class="section">
        <div class="section-title">Additional Resources</div>
        <div class="content-block">
    """
    for res in resources:
        name = res.get("name", "N/A")
        contact = res.get("contact", "N/A")
        website = res.get("website", "N/A")
        html_content += f"""
            <div class="resource-item">
              <p><span class="bold">{name}</span></p>
              <p>Contact: {contact}</p>
              <p>Website: {website}</p>
            </div>"""
    
    html_content += f"""
        </div>
      </div>

      <!-- Conclusion -->
      <div class="section">
        <div class="section-title">Conclusion</div>
        <div class="content-block">
          <p>{conclusion}</p>
        </div>
      </div>

      <!-- Disclaimer -->
      <div class="disclaimer">
        <p>{disclaimer}</p>
      </div>

      <!-- Footer -->
      <div class="footer">
        <p>Generated by Instalysis Mental Health Assessment System</p>
        <p>This is an AI-generated report and should be reviewed by a qualified mental health professional</p>
      </div>

    </body>
    </html>
    """
    return html_content

###############################################################################
# Helper: Create PDF from HTML using WeasyPrint
###############################################################################
def create_pdf_from_html(html_content):
    pdf_file = HTML(string=html_content).write_pdf()
    return pdf_file

###############################################################################
# Sidebar Navigation: Analysis, Chat, or Report
###############################################################################
page_options = ["Analysis", "Chat", "Report"]

# Initialize session state keys safely
if "current_page" not in st.session_state:
    st.session_state.current_page = "Analysis"

if "page_redirect_to" not in st.session_state:
    st.session_state.page_redirect_to = None

# Handle programmatic page change BEFORE sidebar renders
if st.session_state.page_redirect_to:
    st.session_state.current_page = st.session_state.page_redirect_to
    st.session_state.page_redirect_to = None
    st.rerun()

# Render sidebar navigation (widget owns this key only)
selected_page = st.sidebar.radio("Navigation", page_options, index=page_options.index(st.session_state.current_page))

# Manual selection via sidebar
if selected_page != st.session_state.current_page:
    st.session_state.current_page = selected_page

current_page = st.session_state.current_page

###############################################################################
# Analysis Page
###############################################################################
if current_page == "Analysis":
    st.title("Instalysis - Instagram Analysis")
    st.subheader("Enter Instagram Username")
    
    with st.form("analysis_form"):
        username_input = st.text_input("Instagram Username:")
        submit_button = st.form_submit_button("Analyze")
    
    if submit_button and username_input.strip():
        st.session_state.username = username_input.strip()
        async def run_fetch_captions():
            with st.spinner("Fetching captions..."):
                return await fetch_captions(username_input)
        captions = asyncio.run(run_fetch_captions())
        if captions:
            st.success(f"Fetched {len(captions)} captions.")
            cleaned_captions = preprocess_captions(captions)
            st.write(f"**Cleaned Captions (first 10):** {cleaned_captions[:10]}")
            cleaned_captions = [c for c in cleaned_captions if c.strip()]
            if not cleaned_captions:
                st.warning("No valid captions to analyze.")
            else:
                predicted_state = predict_state_of_mind(cleaned_captions)
                st.session_state.user_mood = predicted_state
                st.success(f"Predicted Mental Health Condition: {predicted_state}")
                
                # Get recommendations from Gemini
                recs = get_recommendations(predicted_state)
                st.session_state.recommendations = recs
                st.markdown("**Recommendations:**")
                st.markdown(recs)
                
                st.session_state.ig_analysis_done = True
                st.info("Analysis complete! Switching to the Chat page...")
                st.session_state.page_redirect_to = "Chat"
                st.rerun()
        else:
            st.warning("No captions returned. Verify the username or account privacy.")

###############################################################################
# Chat Page
###############################################################################
elif current_page == "Chat":
    st.title("Instalysis - AI Counsellor Chat")
    
    if not st.session_state.ig_analysis_done:
        st.warning("Please complete Instagram Analysis first (switch to the Analysis page).")
    else:
        st.subheader("Chat with your AI Counsellor")
        
        # If no chat messages exist, set an initial bot message using recommendations.
        if not st.session_state.messages:
            username = st.session_state.username
            user_mood = st.session_state.user_mood
            recs = st.session_state.recommendations
            initial_msg = (
                f"Hi @{username}, I saw that you just analysed your mental state and it appears you're feeling {user_mood}. "
                f"Here are some recommendations to help you: {recs}. "
                "I am here to listen—please feel free to share anything with me."
            )
            st.session_state.messages = [{"role": "bot", "content": initial_msg}]
        
        # Display chat messages
        from streamlit_chat import message  # ensure import in this scope
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "bot":
                message(msg["content"], is_user=False, key=f"bot_{i}")
            else:
                message(msg["content"], is_user=True, key=f"user_{i}")
        
        # Chat input area
        user_input = st.text_input("Type your message...", key="chat_input", placeholder="Enter your message...", label_visibility="hidden")
        send_btn = st.button("Send", key="send_btn")
        if send_btn and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Generating response..."):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(gemini_chatbot_logic, st.session_state.messages, user_input)
                    bot_response = future.result()
            st.session_state.messages.append({"role": "bot", "content": bot_response})
            st.rerun()
        
        # Bottom row: Reset Chat and Generate Report buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset Chat", key="reset_btn"):
                st.session_state.messages.clear()
                st.session_state.ig_analysis_done = False
                st.session_state.user_mood = None
                st.session_state.recommendations = ""
                st.rerun()
        with col2:
            if st.button("Generate Comprehensive Report", key="report_btn"):
                with st.spinner("Generating report..."):
                    report_json_string = get_comprehensive_report(st.session_state.messages, st.session_state.username)
                
                report_dict = None
                report_html = ""
                error_message = ""
                
                # Attempt to extract JSON string, looking for ```json ... ``` pattern first
                try:
                    json_match = re.search(r'```json\s*(.*?)\s*```', report_json_string, re.DOTALL)
                    if json_match:
                        extracted_json_string = json_match.group(1)
                        report_dict = json.loads(extracted_json_string)
                    else:
                        # If no markdown json block, try parsing the whole string
                        report_dict = json.loads(report_json_string)
                        
                    if report_dict:
                        # HTML formatting function:
                        report_html = format_report_as_html(report_dict.get("report", {}))
                    else:
                        error_message = "Could not extract or parse JSON report data."
                        st.error(error_message)
                        logger.error(f"Failed to extract/parse JSON:\n{report_json_string}")

                except json.JSONDecodeError as e:
                    error_message = f"Failed to decode JSON from report data: {e}"
                    st.error(error_message)
                    logger.error(f"JSON Decode Error: {e}\nRaw response:\n{report_json_string}")
                except Exception as e:
                    error_message = f"An unexpected error occurred during report processing: {e}"
                    st.error(error_message)
                    logger.error(f"Unexpected error during report processing: {e}\nRaw response:\n{report_json_string}")

                # Store the generated HTML or an error message to session state
                if report_html:
                    st.session_state.report_html = report_html
                    st.session_state.page_redirect_to = "Report"
                    st.rerun()
                elif error_message:
                    # Store the error message to be displayed on the report page if needed
                    # Or just let the st.error handle it on the chat page before redirecting
                    pass # The error is already shown via st.error
                else:
                     # Fallback if no html generated and no specific error message
                    st.error("Could not generate the report in the expected format.")
                    logger.error(f"Report generation failed without specific error message.\nRaw response:\n{report_json_string}")

###############################################################################
# Report Page
###############################################################################
elif current_page == "Report":
    st.title("Instalysis - Comprehensive Report")
    if st.session_state.report_html:
        st.markdown(st.session_state.report_html, unsafe_allow_html=True)
        pdf_bytes = create_pdf_from_html(st.session_state.report_html)
        st.download_button(
            label="Download Report as PDF",
            data=pdf_bytes,
            file_name="Medical_Report.pdf",
            mime="application/pdf"
        )
    else:
        st.info("No report available. Please generate a report from the Chat page.")

###############################################################################
# Disclaimer Footer (Shown on all pages)
###############################################################################
st.markdown("---")
st.markdown("""
<div style="font-size: 13px; color: #555;">
<strong>Disclaimer:</strong> This AI Counsellor is for supportive conversation only and is not a substitute for professional mental health care. 
Please consult a licensed professional for personalized advice.
</div>
""", unsafe_allow_html=True)
