# import os
import streamlit as st
# import numpy as np
# from PIL import  Image

from Pages import Perform_app, Scoring_loans

# Custom imports 
from Multipages.Multi_pages import MultiPage

# Create an instance of the app 
st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 250px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 500px;
            margin-left: -500px;
            }
            </style>
            """,
            unsafe_allow_html=True)

app = MultiPage()

app.add_page("Performance Model", Perform_app.perform)
app.add_page("Interpretation Model", Scoring_loans.explainer_model)

# The main app
app.run()



