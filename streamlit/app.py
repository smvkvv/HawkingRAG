import streamlit as st
import requests
import time

st.set_page_config(page_title="Hawking RAG", page_icon="🌌")
st.title("Hawking RAG")

st.markdown(
    """
    <style>
    [data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] p {
        text-align: left !important;
    }
    [data-testid="stChatMessage"].user div[data-testid="stMarkdownContainer"] p {
        text-align: right !important;
    }

    [data-testid="stChatMessage"].user .stMarkdown .stIcon svg {
        float: right !important;
    }
    [data-testid="stChatMessage"].assistant .stMarkdown .stIcon svg {
        float: left !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

def send_message_to_rag(user_input: str) -> dict:
    api_url = "http://interface:8000/ask/"
    payload = {"question": user_input}
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()

        response_json = response.json()
        message = response_json.get("response", "No valid response from RAG.")
        contexts = response_json.get("contexts", "No valid contexts from RAG")

        if type(contexts) is list:
            chapters = list(set([context['chapter'] for context in contexts]))
            return {"response": message, "chapters": chapters}

        return {"response": message}
    except requests.exceptions.RequestException:
        return {"response": "Error: Could not reach the backend."}


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Напишите свой вопрос..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        partial_response = ""
        full_response = send_message_to_rag(user_input)
        for word in full_response['response'].split():
            partial_response += word + " "
            msg_placeholder.markdown(partial_response)
            time.sleep(0.04)

        if 'chapters' in full_response and len(full_response["chapters"]) != 0:
            partial_response += "\n\n\nДля ответа использовалась информация из следующих глав:"
            msg_placeholder.markdown(partial_response)
            time.sleep(0.04)

            for chapter in full_response["chapters"]:
                partial_response += f"\n\n{chapter}"
                msg_placeholder.markdown(partial_response)
                time.sleep(0.04)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
