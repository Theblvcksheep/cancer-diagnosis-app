import streamlit as st
import pickle as pkl
import pandas as pd


def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.write("Hello world")

if __name__ == '__main__':
    main()
