import helpers as lch_helper
import streamlit as st
import textwrap

st.title("Your Youtube Assitant")

with st.sidebar:
    with st.form(key="my_form"):
        youtube_url = st.sidebar.text_area(
            label="Please provide your youtube url", max_chars=50
        )
        query = st.sidebar.text_area(
            label="Ask me anything about the video", max_chars=55, key="query"
        )
        openai_api_key = st.sidebar.text_input(
            label="OpenAI API Key",
            key="langchain_search_api_key_openai",
            max_chars=100,
            type="password",
        )
        submit_button = st.form_submit_button(label="Submit")

if query and youtube_url:
    if lch_helper.is_valid_youtube_video(youtube_url):
        st.error("This video is unavailable please enter a valid youtube url")
        st.stop()
    if not openai_api_key or not lch_helper.check_openai_api_key(openai_api_key):
        st.warning("Please add your OpenAI API key to continue or Wrong API Key.")
        st.stop()
    else:
        db = lch_helper.create_vector_db(youtube_url, openai_api_key)

        response, documents = lch_helper.get_response_from_query(
            db, query, openai_api_key
        )
        st.subheader("Answer:")
        st.text(textwrap.fill(response, width=85))
