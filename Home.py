import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="XtraHelp Analytics",
)

# Display the title and welcome message
st.markdown("<h1 style='text-align: center;'>XtraHelp Analytics</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Welcome!</h2>", unsafe_allow_html=True)

# Description section
st.markdown("<h3>Safeguarding people with instant access to reliable information</h3>", unsafe_allow_html=True)
st.markdown(
    "<h3>Explore your dataset through four powerful analytical approaches</h3>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4>To begin, upload your dataset from the sidebar and click on <b>Show Analysis</b></h4>",
    unsafe_allow_html=True
)
