import streamlit as st
import requests

st.set_page_config(page_title="Aviation RAG System", page_icon="✈️", layout="centered")

API_URL = "http://127.0.0.1:8000/ask"

# -------------- UI TITLE AND INPUT ----------------
st.title(" Aviation RAG Assistant")
st.write("Ask anything based on CPL/ATPL Aviation Manuals.")

query = st.text_input("Your Question:", placeholder="e.g., What is dew point temperature?")

debug = st.checkbox("Show retrieved context", value=False)

if st.button("Submit") and query:
    with st.spinner("Searching documents & generating response..."):
        try:
            response = requests.get(API_URL, params={"q": query, "debug": debug}, timeout=120)
            data = response.json()

            answer = data.get("answer", " No answer returned")
            citations = data.get("citations", [])
            context = data.get("context_used", None)

            # ----------- OUTPUT RESPONSE ----------------
            st.subheader(" Answer")
            st.write(answer)

            st.subheader(" Citations (Sources)")
            if citations:
                for c in citations:
                    st.write("•", c)
            else:
                st.write(" No citations found.")

            if debug and context:
                st.subheader(" Retrieved Context")
                with st.expander("View context"):
                    st.write(context)

        except Exception as e:
            st.error(f" API Error: {e}")
