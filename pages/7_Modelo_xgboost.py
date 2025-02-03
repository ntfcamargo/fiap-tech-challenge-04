import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import datetime as dt
import yfinance as yf

# Função para calcular métricas
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mae, mse, rmse, mape

# Função para criar features temporais
def create_time_features(df):
    df['Ano'] = df.index.year
    df['Mês'] = df.index.month
    df['Dia'] = df.index.day
    df['Dia_Semana'] = df.index.weekday
    return df

# Carregar os dados (substituir pelo seu DataFrame real)
@st.cache_data
def load_data():
    # Baixar os dados do petróleo
    df = yf.Ticker("BZ=F")

    # Obter histórico (personalize o período e intervalo)
    df = df.history(period="max", interval="1d")  # Histórico diário completo
    df = df.reset_index()
    df['Date'] = df['Date'].dt.tz_localize(None)
    df.rename(columns={'Close':'Preço - petróleo bruto (Brent) - em dólares'},inplace=True)
    df = df[['Date','Preço - petróleo bruto (Brent) - em dólares']]
    df = create_time_features(df)
    FEATURES_ALL = ['Ano', 'Mês', 'Dia', 'Dia_Semana']
    TARGET = "Preço - petróleo bruto (Brent) - em dólares"
    df = df.dropna()  # Remover valores nulos gerados pela média móvel
    return df

# Interface Streamlit
st.title("Predição do Preço do Petróleo com XGBoost")

# Baixar os dados do petróleo
df = yf.Ticker("BZ=F")

# Obter histórico (personalize o período e intervalo)
df = df.history(period="max", interval="1d")  # Histórico diário completo
df = df.reset_index()
df['Date'] = df['Date'].dt.tz_localize(None)
df.rename(columns={'Close':'Preço - petróleo bruto (Brent) - em dólares'},inplace=True)
df = df[['Date','Preço - petróleo bruto (Brent) - em dólares']]
df = df.set_index('Date')
FEATURES_ALL = ['Ano', 'Mês', 'Dia', 'Dia_Semana']
TARGET = "Preço - petróleo bruto (Brent) - em dólares"

# Carregar dados
basef = create_time_features(df)

# Seleção de features e número de dias futuros
FEATURES_ALL = ['Ano', 'Mês', 'Dia', 'Dia_Semana']
TARGET = "Preço - petróleo bruto (Brent) - em dólares"

selected_features = st.multiselect("Escolha as Features", FEATURES_ALL, default=FEATURES_ALL)
diaspred = st.slider("Selecione o número de dias futuros", min_value=10, max_value=365, value=50, step=5)

if st.button("Rodar Modelo"):
    # Separação dos dados
    x_train, y_train = basef[selected_features], basef[TARGET]

    # Treinamento do modelo
    reg = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=300, learning_rate=0.1)
    reg.fit(x_train, y_train)

    # Criar datas futuras
    last_date = basef.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=diaspred, freq='D')

    # Criar DataFrame para previsões futuras
    future_df = pd.DataFrame(index=future_dates)


    # Criar features temporais futuras
    future_df = create_time_features(future_df)
    # Avaliação do modelo nos últimos x dias conhecidos
    train_size = basef.shape[0] - diaspred
    train, test = basef.iloc[:train_size],basef.iloc[train_size:]
    x_train,y_train = train[FEATURES_ALL],train[TARGET]
    x_test,y_test = test[FEATURES_ALL],test[TARGET]
    preds_test = reg.predict(x_test)
    metrics_xgb = calculate_metrics(y_test, preds_test)

    # Exibir métricas
    st.subheader("Métricas do Modelo")
    st.write(f"**MAE:** {metrics_xgb[0]:.2f}")
    st.write(f"**MSE:** {metrics_xgb[1]:.2f}")
    st.write(f"**RMSE:** {metrics_xgb[2]:.2f}")
    st.write(f"**MAPE:** {metrics_xgb[3]:.2f}%")

    #FUTURO
    preds = reg.predict(future_df)
    future_df['y_hat'] = preds
    # Plotar gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(basef.index[-diaspred:], basef[TARGET].iloc[-diaspred:], label='Preço Real (Brent)', color='green')
    ax.plot(future_df.index, future_df['y_hat'], linestyle='--', label='Previsões (Futuro)', color='blue')
    ax.set_title("Comparação de Previsões e Valores Reais do Brent")
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço (em dólares)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Exibir DataFrame de previsões futuras
    st.subheader("Valores Previstos")
    st.dataframe(future_df)
