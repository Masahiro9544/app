import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

path = os.getcwd()

st.set_page_config(layout="wide")

st.header('従業員の離職予測')
st.subheader(':blue[Employee Attritioin Predict]')

col1, col2, col3 = st.columns(3)

with col1:
    #Age
    number1 = st.number_input('年齢', value=30, step=1)
    #Gender
    number2 = st.selectbox(
        '性別(0が男性、1が女性)', [0, 1]
    )
    #Distance from Home
    number10 = st.number_input('職場までの距離(km)',value=0, step=1)
    #Marital Status_Divorced
    number11 = st.selectbox(
        '婚姻状況(離婚している場合は1)', [0, 1]
    )
    #Marital Status_Married
    number12 = st.selectbox(
    '婚姻状況(結婚している場合は1)', [0, 1]
        )
    #Marital Status_Single
    number13 = st.selectbox(
        '婚姻状況(独身の場合は1)', [0, 1]
    )
    #Number of Dependents
    number14 = st.number_input('扶養家族数', value=0, step=1)

with col2:
    #Years at Company
    number3 = st.number_input('在籍年数（年）', value=10, step=1)
    #Monthly Income
    number4 = st.number_input('月収(ドル)', value=20, step=1)
    #Work-Life Balance
    number5 = st.selectbox(
        'ワークライフバランス（0:悪い~3:非常に良い）', [0, 1, 2, 3]
    )
    #Job Satisfaction
    number6 = st.selectbox(
        '仕事満足度（0:悪い~3:非常に良い）', [0, 1, 2, 3]
    )
    #Performance Rating
    number7 = st.selectbox(
        '業績評価（0:悪い~3:非常に良い）', [0, 1, 2, 3]
    )
    #Number of Promotions
    number8 = st.number_input('昇進回数',value=0, step=1)
    #Overtime
    number9 = st.selectbox(
        '残業の有無(0:無し、1:有り)', [0, 1]
    )
    #Job Level
    number15 = st.selectbox(
        '役職（0:Entry、1:Middle、2:Senior）', [0, 1, 2]
    )
    #Company Tenure
    number16 = st.number_input('企業年数', value=0, step=1)

with col3:
    #Remote Work
    number17 = st.selectbox(
        'リモートの有無(0:無し、1:有り)', [0, 1]
    )
    #Leadership Opportunities
    number18 = st.selectbox(
        'リーダーシップの機会(0:無し、1:有り)', [0, 1]
    )
    #Innovation Opportunities
    number19 = st.selectbox(
        'イノベーションの機会(0:無し、1:有り)', [0, 1]
    )
    #Company Reputation
    number20 = st.selectbox(
        '会社の評価（0:悪い~3:非常に良い）', [0, 1, 2, 3]
    )
    #Employee Recognition
    number21 = st.selectbox(
        '従業員評価（0:悪い~3:非常に良い）', [0, 1, 2, 3]
    )

input_data = {
    'Age': [number1],
    'Gender': [number2],
    'Years at Company': [number3],
    'Monthly Income': [number4],
    'Work-Life Balance': [number5],
    'Job Satisfaction': [number6],
    'Performance Rating': [number7],
    'Number of Promotions': [number8],
    'Overtime': [number9],
    'Distance from Home': [number10],
    'Marital Status_Divorced': [number11],
    'Marital Status_Married': [number12],
    'Marital Status_Single': [number13],
    'Number of Dependents': [number14],
    'Job Level': [number15],
    'Company Tenure': [number16],
    'Remote Work': [number17],
    'Leadership Opportunities': [number18],
    'Innovation Opportunities': [number19],
    'Company Reputation': [number20],
    'Employee Recognition': [number21]
}

input_df = pd.DataFrame(input_data)

model = joblib.load(f'{path}/model.pkl')
predictions = model.predict_proba(input_df)
pred1 = predictions[0, 1]
pred = pred1 * 100

if pred1 >= 0.75:
    risk = '大'
elif pred1 >= 0.5:
    risk = '中'
else:
    risk = '小'

st.markdown(f'離職率は **:red[{pred:.1f}]** %です')
st.markdown(f'離職リスクは **:red[{risk}]**')