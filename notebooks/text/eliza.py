import re
import time
from collections.abc import Generator

import streamlit as st


def replace_do_you_questions(text: str) -> str:
  pattern = r"^Do you (.*?)\??$"
  return re.sub(pattern, r"I \1.", text, flags=re.IGNORECASE)


def respond(text: str) -> None:
  with st.chat_message("user"):
    st.markdown(text)

  st.session_state["messages"].append({"role": "user", "content": text})

  text = text.strip()
  response = replace_do_you_questions(text)

  def stream_response() -> Generator[str]:
    for word in response.split(" "):
      yield word + " "
      time.sleep(0.1)

  with st.chat_message("assistant"):
    st.write_stream(stream_response)

  st.session_state["messages"].append({"role": "assistant", "content": response})


if "messages" not in st.session_state:
  st.session_state["messages"] = []

for message in st.session_state["messages"]:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

if prompt := st.chat_input("Type something to chat, or ask me a question!"):
  respond(prompt)
