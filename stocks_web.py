# Streamlit Stock App

#Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import base64

#Page title and icon
st.set_page_config(page_title='Stocks Graphs', page_icon='chart_with_upwards_trend')

#Scraping data
@st.cache
def load_sp_data():
    url= 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0)
    sp_df = html[0]
    return sp_df

st.write("""
Stock Trends for 2019
""")


sp_df = load_sp_data()

#grouping data by sector
sector = sp_df.groupby('GICS Sector')

#sidebar
sort_sector_unq = sorted(sp_df['GICS Sector'].unique())
select_sector = st.sidebar.multiselect('Sector', sort_sector_unq)

#filter data
df_select_sector = sp_df[(sp_df['GICS Sector'].isin(select_sector))]

st.header('Display Companies in Selected Sector')
st.write('Data Dimension: ' + str(df_select_sector.shape[0]) + ' rows and ' + str(df_select_sector.shape[1]) + ' columns.')
st.dataframe(df_select_sector)

#donwload s%p500
def downloadfile(sp_df):
    csv = sp_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() #bites convertion
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href

st.markdown(downloadfile(df_select_sector), unsafe_allow_html=True)


#yfinance data
data = yf.download(
        tickers = list(df_select_sector[:10].Symbol),
        period = "ytd",
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )





st.sidebar.header('Chose Something')