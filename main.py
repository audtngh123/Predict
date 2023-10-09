import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import os
from matplotlib import font_manager as fm
font_path = "./ngulim.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

st.set_option('deprecation.showPyplotGlobalUse', False)

def machineLearning(df):

    x = df['OPS']
    y = df['Sal']
    X=x.values.reshape(-1,1)
    Model = LinearRegression()
    Model.fit(X, y)
    y_pred = Model.predict(X)

    return Model, y_pred, y

def predictFunc(Model, dataList):
    result = Model.predict([dataList])
    return result

st.title('OPS를 이용한 선수가치 예측하기')

df = pd.read_csv('./playerstat.csv')

tab1, tab2, tab3 = st.tabs(['Home', '프로젝트 결과', '프로젝트 설명'])

with tab1:
    st.info('예측은 프로젝트 결과에서, 이론적 부분은 프로젝트 설명에 있습니다.')

with tab2:
    if st.checkbox('데이터 보기'):
        st.write(df)
        dummy1, y_pred, y=machineLearning(df)
        fig=plt.figure
        plt.scatter(df['OPS'], y)
        plt.title('OPS에 따른 선수가치 예측')
        plt.xlabel('OPS')
        plt.ylabel('연봉(단위:만)')
        plt.plot(df['OPS'], y_pred, color='red')
        st.pyplot()

    if st.checkbox('예측하기'):
        Model, y_pred, y_test = machineLearning(df)
        with st.form('Form data'):
            OPS = st.number_input('OPS', min_value=0.000, max_value=4.000)
            if st.form_submit_button('확인'):
                result = predictFunc(Model, [OPS])
                st.write(f'예상되는 연봉 : {result.round(2)}만원')

with tab3:
    st.write("대한민국의 야구는 축구, 농구, 배구와 함께 국내 4대 프로 스포츠로서, 시장규모만 봐도 2위인 축구와 약 7배 차이가 나는 자타가 공인하는 대한민국 최고의 인기 스포츠이다.")
    st.write("입장수익을 기준으로 야구는 730억 9000만, 축구는 110억 7100만, 농구는 64억 7000만, 배구는 20억 5000만이다. 그런데 대한민국의 야구는 MLB와 비교해 선수들의 낮은 수준에 들어가는 비용이 많다고 평가되며 수익률은 형편없는 편이다.")
    st.write("따라서 우리는 야구에 사회과학의 게임 이론과 통계학적 방법론을 적극적으로 도입해 기존 한국 야구 기록의 부실한 부분은 보완하고 무엇보다 선수의 가치에 대해 깊이 있는 접근을 시도하려 한다.") 
    st.write("기존의 관습적 선수 평가론을 부정하고 야구 선수의 가치 평가에 대해 좀 더 과학적, 계량적인 평가를 하고자 한다. 다만 시간적 제약 때문에 이 프로그렘에서 분석하는 선수들은 타자로 한정한다.")
    st.write("타율보다는 실질적으로 타자의 생산성을 잘 표현하면서, 계산하기 쉽고(SLG+OBP), 여러 관점에서 적합한 결론을 돌출할 수 있다는 점에서 OPS를 지표로 삼았으며, 평소에 비해 과대평가 & 과소평가된 선수를 확인할 수 있다.")
    st.write("따라서 이 프로젝트에서는 OPS를 이용하여 OPS와 타자의 시장가치간 관계를 통해 평균적으로 타자의 OPS와 시장가치의 계수를 기준으로 프로젝트를 설계하였다.")
