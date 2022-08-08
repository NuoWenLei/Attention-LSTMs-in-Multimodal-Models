import streamlit as st
import base64

st.set_page_config(
	page_title = "GIF Viz"
)

st.header("Attention Bottleneck LSTM Predictions vs. Truth")

file_ = open("infection_maps_true_v_pred_attention.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<img style = "max-width: 100%" src="data:image/gif;base64,{data_url}" alt="attention gif">',
    unsafe_allow_html=True
)

st.header("Late Fusion LSTM Predictions vs. Truth")

file_ = open("infection_maps_true_v_pred_lstm.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<img style = "max-width: 100%" src="data:image/gif;base64,{data_url}" alt="lstm gif">',
    unsafe_allow_html=True
)
