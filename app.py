import streamlit as st
import tensorflow as tf
from keras.models import load_model

st.title("Spamify")
st.write("Classify messages as spam or legitimate using a trained deep learning model.")
st.markdown("_Built with TensorFlow and Streamlit_")
st.write("")
st.write("")
st.write("")
st.write("")

@st.cache_resource
def initialize():
    return load_model("spam_classifier.keras")


st.markdown("##### Enter message: ")
user_input = st.text_input("", label_visibility="collapsed")

if st.button("Classify"):
    model = initialize()
    if not user_input.strip():
        st.error("Please enter a valid message.")
    else:
        user_input = tf.constant([user_input])
        prediction = model.predict(user_input)
        confidence = float(prediction[0][0])
        if confidence >= 0.5:
            st.error(f"This is {confidence:.0%} spam")
        else:
            st.success(f"This is {(1-confidence):.0%} legitimate")
