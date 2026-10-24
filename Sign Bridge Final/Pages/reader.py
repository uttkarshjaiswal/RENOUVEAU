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

left, center, right = st.columns([1, 7, 1])
with right:
    if st.button("HOME", width="content"):
        st.success("loading")
        st.switch_page("1.py")
st.markdown(
    """
    <p style="
        font-family:Comic Sans MS;
        font-size: 44px;
        color:white;
    ">
      How to use?
    """,
    unsafe_allow_html=True)

st.markdown(
    """
    <p style="
        font-family:Comic Sans MS;
        font-size: 24px;
        color:white;
    ">
      1. press the convert button to start the camera and text section  <br>
      2. To convert the text into speech press speak button when camera starts <br>
      3. To record a custom gesture, click RECORD and enter its label and sample count <br>
      4. Retrain the model after recording so it can recognize the new gesture <br>
      5. Video will be exported as needed with subtitles when you press the export button after your work is done <br>
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
left, center, right = st.columns([1, 1, 1])
with left:
    if st.button("CONVERT", width=200):
        st.success("loading")
        st.switch_page("Pages/app.py")
        
with right:
    if st.button("RECORD", width=200):
        st.success("loading")
        st.switch_page("Pages/record.py")
