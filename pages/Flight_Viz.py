import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import date

flight_df = pd.read_csv("flight_filtered.csv")

st.set_page_config(
	page_title = "Flight Viz"
)

inp_date = st.date_input(
	label = "Flight Graph Date (2020/01/01 - 2020/03/31)",
	value = date(year = 2020, month = 6, day = 12),
	min_value = date(year = 2020, month = 1, day = 1),
	max_value = date(year = 2022, month = 3, day = 31)
)

if inp_date is not None:
	formatted_inp_date = inp_date.strftime("%Y/%m/%d")

	df_sample = flight_df[flight_df["date_adjusted"] == formatted_inp_date]

	G = nx.from_pandas_edgelist(df_sample, source = "state_from", target = "state_to", edge_attr = "count_logged")
	edge_weights = [i['count_logged'] for i in dict(G.edges).values()]
	labels = {i:i for i in dict(G.nodes).keys()}
	fig, ax = plt.subplots(figsize=(12,5))
	pos = nx.spring_layout(G, seed = 100)
	nx.draw_networkx_nodes(G, pos, ax = ax)
	nx.draw_networkx_edges(G, pos, width=edge_weights, ax=ax)
	_ = nx.draw_networkx_labels(G, pos, labels, ax=ax)

	st.pyplot(fig = fig, clear_figure = True)




