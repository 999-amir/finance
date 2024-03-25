# ============================================================================== packages
# base packages
import pandas as pd
import numpy as np
# visualization
import plotly.graph_objects as go
import plotly.express as px
# collect data
import yfinance as yf
# preprocessing
from sklearn.preprocessing import StandardScaler
# ML
from xgboost import XGBRegressor
# deploy
import streamlit as st
# ============================================================================== input
st.title('yahoo finance prediction')
stock_name = st.text_input('add yahoo finance symbol:', 'ETH-USD')
limit_precent = st.slider("select chart limit %", 1, 100, 100)/100
show_train_prediction = st.checkbox('show train process', True)
predict_future = st.checkbox('show future', True)
# ============================================================================== download dataset
df = yf.Ticker(stock_name).history(period='max').reset_index()
df = df[:int(len(df) * limit_precent)]
# ============================================================================== train & test split data
split_date = str(df['Date'].iloc[int(len(df) * 0.9)]).split(' ')[0]
# print('date of start prediction:', split_date)
# ============================================================================== get ready dataset
# create feature
df['quarter'] = df['Date'].dt.quarter
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['dayofyear'] = df['Date'].dt.dayofyear
df['weekofyear'] = df['Date'].dt.isocalendar().week
# train & test
if predict_future:
    # future dataframe
    future = pd.date_range(start=str(df['Date'][-1:].values[0]).split('T')[0], periods=int(len(df) * 0.1), freq='d')
    print(f'{stock_name} prediction from {future[0]} to {future[-1]}')
    df_future = pd.DataFrame(future, columns=['Date'])
    df_future['quarter'] = df_future['Date'].dt.quarter
    df_future['month'] = df_future['Date'].dt.month
    df_future['year'] = df_future['Date'].dt.year
    df_future['dayofyear'] = df_future['Date'].dt.dayofyear
    df_future['weekofyear'] = df_future['Date'].dt.isocalendar().week
    train = df
    test = df_future
else:
    train = df[df.Date <= split_date]
    test = df[df.Date >= split_date]
# print('test size:', np.round(len(test) / len(df) * 100, 2), '%')
# print('train size:', np.round(len(train) / len(df) * 100, 2), '%', '\n')
# train & test splited by X & Y
X_train = train[['quarter', 'month', 'year', 'dayofyear', 'weekofyear']]
X_test = test[['quarter', 'month', 'year', 'dayofyear', 'weekofyear']]
# scaling
sc = StandardScaler()
X_train = pd.DataFrame(sc.fit_transform(X_train), columns=['quarter', 'month', 'year', 'dayofyear', 'weekofyear'])
X_test = pd.DataFrame(sc.fit_transform(X_test), columns=['quarter', 'month', 'year', 'dayofyear', 'weekofyear'])
Y_train = train['Open']
if not predict_future:
    Y_test = test['Open']
# ============================================================================== machine learning
xgbr = XGBRegressor(n_estimators=200, learning_rate=0.02, max_depth=3)
xgbr.fit(X_train, Y_train)
open = xgbr.predict(X_test)
# ============================================================================== visualization
# plot prediction
fig1 = px.line(x=train.Date, y=train['Open'], title='train')
fig1.update_traces(line_color='darkgrey')
if predict_future:
    fig3 = px.line(x=test.Date, y=open + train[-1:].Open.values[0] - open[0], title='test predicted')
    fig3.update_traces(line_color='lime')
else:
    fig2 = px.line(x=test.Date, y=test['Open'], title='test')
    fig2.update_traces(line_color='red')
    fig3 = px.line(x=test.Date, y=open + test[:1].Open.values[0] - open[0], title='test predicted')
    fig3.update_traces(line_color='lime')
if show_train_prediction:
    fig4 = px.line(x=train.Date, y=xgbr.predict(X_train), title='train prediction')
    fig4.update_traces(line_color='cyan')
    if predict_future:
        fig5 = go.Figure(data=fig1.data + fig3.data + fig4.data)
    else:
        fig5 = go.Figure(data=fig1.data + fig2.data + fig3.data + fig4.data)
else:
    if predict_future:
        fig5 = go.Figure(data=fig1.data + fig3.data)
    else:
        fig5 = go.Figure(data=fig1.data + fig2.data + fig3.data)
fig5.update_xaxes(title_text="Date")
fig5.update_yaxes(title_text="Open price")
fig5.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=300, title=stock_name)
fig5.layout.template = 'plotly_dark'
# fig5.show()
# ============================================================================== deploy
st.header(f'chart plot')
st.plotly_chart(fig5)

def show_big_chart():
    fig5.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=725, title=stock_name)
    fig5.show()
btn = st.button('show chart in new page', on_click=show_big_chart)
