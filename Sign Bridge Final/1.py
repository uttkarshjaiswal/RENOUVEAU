import streamlit as st
st.set_page_config(page_title="RENOUVEAU", layout="wide")
st.html("""
<style>
.stApp {
    background-image:
        linear-gradient(rgba(0, 0, 0, 0.35), rgba(0, 0, 0, 0.35)),
        url("https://i.pinimg.com/736x/a8/99/a7/a899a7489d4cf51f5db54738d802bf31.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
</style>
""")
st.html("""
<style>
div[data-testid="stButton"] button {
    font-size: 30px;
    font-family:Comic Sans MS ;
    color: white;
    background-color: #6C63FF;
}
</style>
""")
st.html("""
<style>
div[data-testid="stButton"] button p {
    font-size: 28px !important;
    font-family: Comic Sans MS !important;
}
</style>
""")
st.markdown(
    """
    <p style="
        font-family:Comic Sans MS;
        font-size: 64px;
        color:white;
    ">
       Welcome! to our app.
    </p>
    """,
    unsafe_allow_html=True)

st.write("")
st.markdown(
    """
    <p style="
        font-family:Comic Sans MS;
        font-size: 54px;
        color:white;
    ">
       INTRODUCING RENOUVEAU.
    </p>
    """,
    unsafe_allow_html=True)

st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.markdown(
    """
    <p style="
        font-family:Comic Sans MS;
        font-size: 24px;
        color:white;
    ">
       INSTRUCTIONS.
    </p>
    """,
    unsafe_allow_html=True

)

st.markdown(
    """
    <p style="
        font-family:Comic Sans MS;
        font-size: 14px;
        color:white;
    ">
       This app is designed specially for people who are unable to speak.
The app works on the hand gestures which you make. The camera will capture the movements and process them.
subsequently the text which is associated with the gesture will be printed in the text menu.
This app can also capture custom hand gesture, when someone does those gestueres the text assingned to it will be printed.
    </p>
    """,
    unsafe_allow_html=True
)
st.write("")
st.write("")
st.write("")
st.write("")
st.markdown(
    """
    <p style="
        font-family:Comic Sans MS;
        font-size: 24px;
        color:white;
    ">
      Features provided.
    """,
    unsafe_allow_html=True)
st.markdown(
    """
    <p style="
        font-family:Comic Sans MS;
        font-size: 14px;
        color:white;
    ">
      1. Gestures into text  <br>
      2. Can convert the text into voice too  <br>
      3. You can make your custom gestures and save it   <br>
      4. Video will be exported as needed with subtitles  <br>
    """,
    unsafe_allow_html=True)
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")


if st.button("START"):
        st.success("loading")
        st.switch_page("Pages/reader.py")
