import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Título do aplicativo
st.write("""### Modelo de Variação Relativa""")
st.write("""
O gráfico abaixo demonstra a variação de diferentes variáveis econômicas e financeiras ao longo do tempo. Para personalizar a análise, utilize os filtros disponíveis na lateral esquerda, onde você pode selecionar as variáveis de interesse e ajustar o período desejado. Essa funcionalidade permite explorar relações específicas entre os indicadores, identificar padrões e tendências ao longo do tempo, e obter insights relevantes que auxiliam em decisões mais informadas e estratégicas.         
         """)

# Importar os dados
tables = pd.read_html("http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view")
df = pd.DataFrame(tables[2]).drop(0)
df.columns = ['Data', 'Preço - petróleo bruto (Brent) - em dólares']
df['Preço - petróleo bruto (Brent) - em dólares'] = df['Preço - petróleo bruto (Brent) - em dólares'].apply(
    lambda x: float(x[:-2] + '.' + x[-2:])
)
df.rename(columns={'Data': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

# Obter dados do S&P 500
sp500 = yf.Ticker("^GSPC")
s500 = sp500.history(period="max", interval="1d").reset_index()
s500['Date'] = s500['Date'].dt.tz_localize(None)

# Obter dados do ouro
gold = yf.Ticker("IAU")
gold_history = gold.history(period="max", interval="1d").reset_index()
gold_history['Date'] = gold_history['Date'].dt.tz_localize(None)

# Obter dados do DXY
dxy = yf.Ticker("DX-Y.NYB")
dxy_history = dxy.history(period="max", interval="1d").reset_index()
dxy_history['Date'] = dxy_history['Date'].dt.tz_localize(None)

# Mesclar os dados
base = df.merge(s500[['Date', 'Close']], on='Date', how='left').fillna(method='ffill').rename(columns={'Close': 'S&P500'})
base = base.merge(gold_history[['Date', 'Close']], on='Date', how='left').fillna(method='ffill').rename(columns={'Close': 'Gold'})
base = base.merge(dxy_history[['Date', 'Close']], on='Date', how='left').fillna(method='ffill').rename(columns={'Close': 'Índice DXY'})
base['Retorno Diário Petróleo'] = base['Preço - petróleo bruto (Brent) - em dólares'].pct_change().fillna(0)

# Dados adicionais (TASA)
tasa = pd.read_csv('https://raw.githubusercontent.com/ntfcamargo/Base-TECH-CHALLENGE-3/refs/heads/main/Dados%20Hist%C3%B3ricos%20-%20Tadawul%20All%20Share.csv', sep=',')
tasa.rename(columns={'Data': 'Date'}, inplace=True)
tasa['Date'] = pd.to_datetime(tasa['Date'])
tasa['Último'] = tasa['Último'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
base = base[base['Date'] > '2005-12-31'].merge(tasa[['Date', 'Último']], on='Date', how='left').fillna(method='ffill').rename(columns={'Último': 'TASA'})

# Normalizar os dados
cols_to_normalize = ['Preço - petróleo bruto (Brent) - em dólares', 'S&P500', 'Gold', 'Índice DXY', 'TASA']
df_normalized = base.copy()
df_normalized[cols_to_normalize] = base[cols_to_normalize].apply(lambda x: x / x.iloc[0])

# Menu de filtros
st.sidebar.header("Filtros")

# Criar filtro para os tickers
tickers_selecionados = st.sidebar.multiselect(
    "Selecione as variáveis que deseja visualizar",
    options=cols_to_normalize,
    default=cols_to_normalize
)

# Converter as datas para o tipo datetime.date
df_normalized['Date'] = pd.to_datetime(df_normalized['Date'])
min_date = df_normalized['Date'].min().date()
max_date = df_normalized['Date'].max().date()

start_date, end_date = st.sidebar.slider(
    "Selecione o período:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

# Aplicar o filtro de datas
df_filtered = df_normalized[
    (df_normalized['Date'] >= pd.Timestamp(start_date)) &
    (df_normalized['Date'] <= pd.Timestamp(end_date))
]
# divisão
st.divider()

# Criar o gráfico com os tickers selecionados
fig, ax = plt.subplots(figsize=(14, 8))
for col in tickers_selecionados:
    ax.plot(df_filtered['Date'], df_filtered[col], label=col, linewidth=2 if col == 'Preço - petróleo bruto (Brent) - em dólares' else 1.5)

ax.set_title("Variação Relativa das Variáveis ao Longo do Tempo", fontsize=16)
ax.set_xlabel("Ano", fontsize=14)
ax.set_ylabel("Variação Relativa (Base 1.0)", fontsize=14)
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.3)

# Exibir o gráfico no Streamlit
st.pyplot(fig)

st.divider()

st.write("""
 A seleção das variáveis utilizadas no modelo foi realizada com base na sua relevância econômica e financeira, considerando indicadores interconectados e representativos de diferentes setores do mercado global. As variáveis escolhidas foram:        
""")

st.write("""
- **Preço do Petróleo Bruto (Brent):** Este indicador é amplamente utilizado como referência global para o preço do petróleo bruto. Suas flutuações refletem a dinâmica da oferta e demanda de energia, além de impactos geopolíticos e políticas climáticas globais.
- **S&P500:** Representa o desempenho das 500 maiores empresas listadas nos Estados Unidos, funcionando como um termômetro da saúde econômica global. Este índice reflete a confiança dos investidores no mercado de capitais.
- **Índice DXY (Dollar Index):** Mede a força do dólar americano em relação a uma cesta de moedas importantes. Este índice é relevante para o estudo de mercados de commodities, pois o dólar impacta diretamente preços internacionais, como os do petróleo e do ouro.
- **Ouro:** Considerado um ativo de reserva de valor e um “porto seguro” em períodos de instabilidade econômica. Suas flutuações apresentam, frequentemente, uma relação inversa com a variação do dólar.
- **TASI (Tadawul All Share Index):** é o principal índice da Bolsa de Valores da Arábia Saudita, a Saudi Exchange (Tadawul). Ele acompanha o desempenho de todas as ações listadas no mercado saudita, sendo um indicador chave da economia local e da saúde do setor empresarial do país. Como a Arábia Saudita é um dos maiores exportadores de petróleo do mundo, o TASI reflete não apenas as dinâmicas internas, mas também tendências globais de energia e investimentos. É amplamente observado por investidores globais devido à sua relevância no mercado de petróleo e às reformas econômicas sauditas.
                          """)
