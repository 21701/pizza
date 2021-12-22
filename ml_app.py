import streamlit as st
import pandas as pd
import numpy as np
import joblib


def run_ml_app():
    st.subheader('Machine Learning 예측')

    # 1. 유저한테, 데이터를 입력받습니다.
    diameter_slider = st.slider('피자 inch',min_value=8.0,max_value=22.0)
    diameter = diameter_slider

    sauce_radio = ['추가함','추가 안함']
    extra_sauce = st.radio('소스 추가',sauce_radio)
    
    if extra_sauce == '추가함':
        extra_sauce = 1
    else:
        extra_sauce = 0

    cheese_radio = ['추가함','추가 안함']
    extra_cheese = st.radio('치즈 추가',cheese_radio)
    
    if extra_cheese == '추가함':
        extra_cheese = 1
    else:
        extra_cheese = 0

    mushrooms_radio = ['추가함','추가 안함']
    extra_mushrooms = st.radio('버섯 추가',mushrooms_radio)
    
    if extra_mushrooms == '추가함':
        extra_mushrooms = 1
    else:
        extra_mushrooms = 0

    size_list = ['XL','Jumbo','Large','Medium','Reguler','Small']
    size = st.selectbox('크기를 선택하세요', size_list)
    
    if size == 'XL':
        xl = 1
        jumbo = 0
        large = 0
        medium = 0
        reguler = 0
        small = 0
    elif size == 'Jumbo':
        xl = 0
        jumbo = 1
        large = 0
        medium = 0
        reguler = 0
        small = 0
    elif size == 'Large':
        xl = 0
        jumbo = 0
        large = 1
        medium = 0
        reguler = 0
        small = 0
    elif size == 'Medium':
        xl = 0
        jumbo = 0
        large = 0
        medium = 1
        reguler = 0
        small = 0
    elif size == 'Reguler':
        xl = 0
        jumbo = 0
        large = 0
        medium = 0
        reguler = 1
        small = 0
    elif size == 'Small':
        xl = 0
        jumbo = 0
        large = 0
        medium = 0
        reguler = 0
        small = 1

    topping_list = ['beef','black papper','chicken','meat','mozzarella','mushrooms',
    'onion','papperoni','sausage','smoked beef','tuna','vegetables']
    topping = st.selectbox('토핑을 선택하세요', topping_list)
    
    
    # 2. 모델에 예측한다.
    # 2-1 신규데이터를 넘파이로 만든다.
    new_data = np.array([diameter,extra_sauce,extra_cheese,extra_mushrooms,xl,jumbo,large,medium,reguler,small])
    new_data = new_data.reshape(1,5)

    # 2-2 스케일러와 인공지능을 변수로 불러온다
    scaler_X = joblib.load('data/scaler_X.pkl')
    scaler_y = joblib.load('data/scaler_y.pkl')
    regressor = joblib.load('data/regressor.pkl')

    # 2-3 신규 데이터를 피쳐스케일링 한다.
    new_data = scaler_X.transform(new_data)
    # 2-4 인공지능에게 예측을 하게 한다.
    y_pred = regressor.predict(new_data)

    # 2-5 예측한 결과는, 다시 원래대로 복구해 줘야 한다.
    print(y_pred)

    y_pred = scaler_y.inverse_transform(y_pred.reshape(1,1))
    print(y_pred)
    
    # 3. 예측 결과를 웹 대시보드에 표시한다.

    btn = st.button('예측 결과 보기')
    # 결과가 소수점으로 나오는데,소수점 뒤 한자리까지만 나오도록
    # 코드 수정하세요
    y_pred = y_pred.round()

    if btn :
        st.write('< 예측 결과 > " 당신의 행복 지수는 {} 입니다. " '.format(y_pred[0,0]))