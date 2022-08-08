import streamlit as st
import numpy as np
import plotly_express as px
import json
from datetime import date

with open("infection_rates.npy", "rb") as infection_npy:
	infection_rates = np.load(infection_npy)

with open("date_to_index.json", "r") as d2i_json:
	date2index = json.load(d2i_json)

state_encoded_abbrev = ['AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC',
'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA',
'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
'VA', 'WA', 'WV', 'WI', 'WY']

st.set_page_config(
	page_title = "Map Viz"
)

infection_range = (infection_rates.min(), infection_rates.max())

inp_date = st.date_input(
	label = "Infection Map Date (2020/01/01 - 2020/03/31)",
	value = date(year = 2020, month = 6, day = 12),
	min_value = date(year = 2020, month = 1, day = 1),
	max_value = date(year = 2022, month = 3, day = 31)
)

if inp_date is not None:
	formatted_inp_date = inp_date.strftime("%Y/%m/%d")


	fig = px.choropleth(
		locations=state_encoded_abbrev,
		locationmode="USA-states",
		color=infection_rates[date2index[formatted_inp_date]],
		scope="usa",
		range_color = infection_range
		)
	
	st.plotly_chart(
		figure_or_data = fig,
		use_container_width = True
		)




