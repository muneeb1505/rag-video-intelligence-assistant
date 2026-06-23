import streamlit as st

from rag_video_assistant import build_chain

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥"
)

st.title("🎥Video Intelligence Assistant")

video_url = st.text_input(
    "Enter YouTube Video URL"
)

if "chain" not in st.session_state:
    st.session_state.chain = None

if st.button("Load Video"):

    if not video_url:
        st.warning("Please enter a YouTube URL.")
        st.stop()

    status = st.empty()
    progress = st.progress(0)

    try:

        def update_progress(message, value):
            status.info(message)
            progress.progress(value)

        st.session_state.chain = build_chain(
            video_url,
            progress_callback=update_progress
        )

        status.success(
            "✅ Video ready! You can now ask questions."
        )

    except Exception as e:
        st.error(f"Error loading video: {e}")

question = st.chat_input(
    "Ask a question about the video..."
)

if question:

    if st.session_state.chain is None:

        st.warning(
            "Load a video first."
        )

    else:

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            st.write_stream(
                st.session_state.chain.stream(
                    question
                )
            )