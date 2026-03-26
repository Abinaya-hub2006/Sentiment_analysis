import streamlit as st
import pandas as pd
import torch
import joblib
import os

from model.lstm_model import SentimentLSTM
from utils.preprocess import clean_text, text_to_sequence
from utils.database import init_db, insert_result, fetch_all
from utils.visualizations import generate_charts
from utils.report import generate_pdf
from utils.email_sender import send_email

st.set_page_config(page_title="AI Dashboard", layout="wide")

init_db()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    word_index = joblib.load("model/word_index.pkl")
    encoder = joblib.load("model/label_encoder.pkl")

    model = SentimentLSTM(vocab_size=5001)
    model.load_state_dict(torch.load("model/model.pth", map_location='cpu'))
    model.eval()

    return model, word_index, encoder

model, word_index, encoder = load_model()

# ---------------- SESSION ----------------
if "df" not in st.session_state:
    st.session_state.df = None

# ---------------- PREDICT ----------------
def predict(text):
    clean = clean_text(text)
    seq = text_to_sequence(clean, word_index)
    tensor = torch.tensor([seq])

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        score, pred = torch.max(probs, dim=1)

    sentiment = encoder.inverse_transform([pred.item()])[0]
    return sentiment, float(score)

# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home","📤 Upload","📊 Results","📈 Plots","📧 Email Report","🗄️ Database"]
)

# ---------------- HOME ----------------
if menu == "🏠 Home":
    st.title("🔥 AI Sentiment Dashboard")
    st.write("Upload → Analyze → Visualize → Email Report")

# ---------------- UPLOAD ----------------
elif menu == "📤 Upload":
    st.title("📤 Upload File")

    file = st.file_uploader("Upload CSV/TXT", type=["csv","txt"])

    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
            texts = df.iloc[:,0].tolist()
        else:
            texts = file.read().decode().split("\n")

        texts = [str(t) for t in texts if str(t).strip() != ""]

        results = []

        with st.spinner("Analyzing..."):
            for t in texts:
                s, sc = predict(t)
                insert_result(t, s, sc, file.name)
                results.append([t, s, sc])

        st.session_state.df = pd.DataFrame(results, columns=["Text","Sentiment","Score"])

        st.success("✅ Analysis Completed!")

# ---------------- RESULTS ----------------
elif menu == "📊 Results":
    st.title("📊 Results")

    if st.session_state.df is not None:
        st.dataframe(st.session_state.df)
    else:
        st.warning("⚠️ Upload file first")

# ---------------- PLOTS ----------------
elif menu == "📈 Plots":
    st.title("📈 Visualizations")

    if st.session_state.df is not None:
        generate_charts(st.session_state.df)

        col1, col2, col3 = st.columns(3)
        col1.image("outputs/images/pie.png", caption="Pie Chart")
        col2.image("outputs/images/bar.png", caption="Bar Chart")
        col3.image("outputs/images/wordcloud.png", caption="WordCloud")
    else:
        st.warning("⚠️ Upload file first")

# ---------------- EMAIL REPORT ----------------
elif menu == "📧 Email Report":
    st.title("📧 Send PDF Report")

    email = st.text_input("Enter Email ID")

    if st.button("Send Report"):

        if st.session_state.df is None:
            st.warning("⚠️ Upload file first")

        elif email.strip() == "":
            st.warning("⚠️ Enter valid email")

        else:
            with st.spinner("Generating & Sending..."):

                # generate report
                generate_charts(st.session_state.df)
                generate_pdf()

                pdf_path = "outputs/reports/report.pdf"

                if not os.path.exists(pdf_path):
                    st.error("❌ PDF not found")
                else:
                    success = send_email(email, pdf_path)

                    if success:
                        st.success("✅ Email sent successfully!")
                    else:
                        st.error("❌ Email failed")

# ---------------- DATABASE ----------------
elif menu == "🗄️ Database":
    st.title("🗄️ Database Records")

    data = fetch_all()

    if len(data) == 0:
        st.warning("No data available")
    else:
        df = pd.DataFrame(data, columns=["id","text","sentiment","score","file"])

        files = df['file'].unique()
        selected = st.selectbox("Select File", files)

        filtered = df[df['file'] == selected]

        st.dataframe(filtered)