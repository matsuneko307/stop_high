import pandas as pd
import numpy as np
import streamlit as st
from pandas_datareader import data
import matplotlib.pyplot as plt
import pickle
import lightgbm as lgb



st.set_page_config(layout="wide")
st.title('過去の似た動きをした銘柄を探すアプリ')
st.write('ストップ高で引けた銘柄が、過去にストップ高で引けたどの銘柄の値動きと似ているかを判定します。')
st.write("------------------------------------------------------------------------------------------------------------------------------------")
st.write('使い方')
st.write('左側に銘柄コードとストップ高で引けた日付(2021/01/01以降)を入力後、判定開始ボタンを押してください。')
st.write('判定開始ボタンを押すと、ストップ高で引けた日付から240営業日前までのデータを使って、その銘柄の値動きと似た動きをした過去の銘柄を探します。')
st.write("------------------------------------------------------------------------------------------------------------------------------------")
st.write("※注意点")
st.write("・上場後250営業日以上経過していない銘柄は判定できません。")
st.write("・2021/01/01以降の日付を入力してください。")
st.write("・入力した銘柄のデータがない場合は、判定できません。")
st.write("・判定には終値を使用しています。")
st.write("・訓練データには2020年末までを使用しています。")
st.write("・一部、予期せぬエラーにより動作しない場合があります。")




st.sidebar.write("証券コードとストップ高の日付を入力")
code = st.sidebar.text_input('1銘柄分の証券コードを半角英数字で入力してください')
end_date = st.sidebar.text_input('2022-01-01のように日付を入力してください')
n = 240
dfy = pd.read_csv("num_stop_high_train.csv")
dfy["up_down"] = np.where(dfy["240"] > dfy["0"], 1, 0)
file = open("num_stop_high_model.pickle", "rb")
gbm = pickle.load(file)
    

        
        
if st.sidebar.button('判定開始') and code and end_date:
    df = data.DataReader(f"{code}.JP", 'stooq', end=end_date).reset_index()
    #エラー処理
    if len(df) < 250:
        st.sidebar.write('※ 上場後250営業日以上経過していません。')
        st.stop()
    if len(df[df["Date"] > "2021-01-01"]) == 0:
        st.sidebar.write('※ 2021/01/01以降の日付を入力してください。')
        st.stop()
    #データフレームの前処理
    st.sidebar.write("グラフはページ下部に表示されますので、スクロールしてください。")
    df = df[df["Volume"] != 0]
    df = df.sort_values("Date", ascending=True)
    df = df.reset_index(drop=True)
    df = df.rename(columns={'Date': 'date'})
    df = df.rename(columns={'Open': 'open'})
    df = df.rename(columns={'High': 'high'})
    df = df.rename(columns={'Low': 'low'})
    df = df.rename(columns={'Close': 'close'})
    df = df.rename(columns={'Volume': 'volume'})
    df = df[df["date"] <= end_date]
    i = df.index[-1]
    #ストップ高日よりn営業日分前までのデータフレームを作成
    df2 = df.iloc[i-n:].copy()
    df2["diff"] = (df2["close"] / df2.loc[df2.index[0], "close"]) - 1
    #予測
    test = df2["diff"].values
    test = test.reshape(1, -1)
    pred = gbm.predict(test)[0]
    #結果の表示
    dfy2 = dfy[(dfy["label"] == pred)].copy()
    #testとdfy2の各行の差分を計算
    for i in dfy2.index:
        dfy2.loc[i, "diff"] = np.sum(np.abs(dfy2.loc[i, "0":"240"] - test[0]))
    dfy2 = dfy2.sort_values("diff", ascending=True)
    #dfy2の上位3行をとtestをグラフで表示
    #ansとtestを同じグラフにプロット
    st.write("------------------------------------------------------------------------------------------------------------------------------------")
    ans = dfy2.iloc[0:3, 0:n+2]
    fig, ax = plt.subplots(figsize=(10, 5))
    #x軸の値を10ごとに表示
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.plot(ans.iloc[0, :-1], label=ans.iloc[0,-1])
    ax.plot(ans.iloc[1, :-1], label=ans.iloc[1,-1])
    ax.plot(test[0], label=code)
    plt.legend()
    st.pyplot(fig)
    st.write('グラフには上位2銘柄と入力した銘柄の値動きを表示し、その他の、上位10銘柄を表で表示します。')
    st.write('グラフのx軸は、240がストップ高発生日、0がその240営業日前を表しています。')
    st.write('グラフのy軸は、0日目からの値動率を表しています。')
    st.write("------------------------------------------------------------------------------------------------------------------------------------")
    st.write("その他の候補")
    st.dataframe((dfy2.loc[dfy2.index[0]: dfy2.index[10], ["code","date"]]))
        

    



else :
  st.write("------------------------------------------------------------------------------------------------------------------------------------")
st.write("アプリの大まかな作成手順")
st.write("1.ストップ高で引けた銘柄を過去240営業日分の終値の値動きを特徴量とし、sklearnのkmeansを使った教師無し分類で10クラスに分類")
st.write("2.ラベル付けしたデータを使って、lightgbmでモデル作成")
st.write("3.モデルを使って、入力した銘柄を分類")
st.write("4.分類したクラスと同じクラスを持つ銘柄の中で、入力した銘柄との値動きの差分が最小の銘柄を上位2銘柄として表示")
st.write("------------------------------------------------------------------------------------------------------------------------------------")

