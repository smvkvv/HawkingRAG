import requests
import streamlit as st


st.set_page_config(page_title="Hawking RAG", page_icon="🌌")
st.title("Hawking RAG")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Function to send message to FastAPI backend
def send_message(user_input: str):
    api_url = "http://interface:8000/ask/"

    payload = {"question": user_input}
    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        return response.json()["response"]
    else:
        return "Error: Could not reach the backend."


# User input text box for chat
user_input = st.text_area("Try to search in Hawking Book:", height=150)

# If user submits a message
if st.button("Send") and user_input:
    # Display the user's message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Send message to backend and get the response
    response_content = send_message(user_input)

    # Display the assistant's response
    st.session_state.messages.append({"role": "assistant", "content": response_content})

# Display the chat history
st.subheader("Chat History")
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"Request: {message['content']}")
    else:
        st.write(f"Response: {message['content']}")