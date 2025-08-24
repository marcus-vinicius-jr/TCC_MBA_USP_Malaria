# -*- coding: utf-8 -*-
"""
Created on Fri May 23 12:50:42 2025

@author: Marcus Vinicius Freire Junior
"""
#%% Instalções necessárias:
# pip install pandas
# pip install numpy
# pip install matplotlib
# pip install seaborn
# pip install scikit-learn
# pip install textstat
# pip install python-Levenshtein
# pip install scipy
# pip install statsmodels
# pip install pingouin
# pip install prince
# pip install plotly
    
#%%Bibliotecas
import pandas as pd
import numpy as np
import os
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from math import pi
from scipy.stats import zscore
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency
from sklearn.metrics import silhouette_score
import pingouin as pg
import prince
import plotly.express as px
import webbrowser

# %% Configuração de caminhos (repositório)
os.makedirs("data", exist_ok=True)     # garante existência (não falha se já houver)
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
#%% Importação das bases de dados
# Caminho da pasta
caminho_pasta = "data"

# Importação com nome da variável correspondente
df_claude        = pd.read_excel(os.path.join(caminho_pasta, "df_claude.xlsx"))
df_copilot       = pd.read_excel(os.path.join(caminho_pasta, "df_copilot.xlsx"))
df_deepseek      = pd.read_excel(os.path.join(caminho_pasta, "df_deepseek.xlsx"))
df_gemini        = pd.read_excel(os.path.join(caminho_pasta, "df_gemini.xlsx"))
df_gpt           = pd.read_excel(os.path.join(caminho_pasta, "df_gpt.xlsx"))
df_gpt_escholar  = pd.read_excel(os.path.join(caminho_pasta, "df_gpt_escholar.xlsx"))
df_gpt_vision    = pd.read_excel(os.path.join(caminho_pasta, "df_gpt_vision.xlsx"))
df_meta_llama    = pd.read_excel(os.path.join(caminho_pasta, "df_meta_llama.xlsx"))
df_reka          = pd.read_excel(os.path.join(caminho_pasta, "df_reka.xlsx"))
df_perplexity    = pd.read_excel(os.path.join(caminho_pasta, "df_perplexity.xlsx"))
df_oms           = pd.read_excel(os.path.join(caminho_pasta, "df_oms.xlsx"))

#%%Lista com DF das IAs
# Lista com todos os DataFrames das IAs
dfs_ia = [
    df_claude,
    df_copilot,
    df_deepseek,
    df_gemini,
    df_gpt,
    df_gpt_escholar,
    df_gpt_vision,
    df_meta_llama,
    df_reka,
    df_perplexity
]
    
#%% Contagem de Palavras + Diferença com a OMS
#Objetivo: 
 # Adicionar uma coluna word_count com o número de palavras da resposta da IA
 # Adicionar uma coluna word_count_diff com a diferença absoluta em relação à resposta da OMS (linha a linha, pelo índice)

def contar_palavras(texto):
    return len(str(texto).split())

# Aplica ao df_oms uma vez
df_oms["word_count"] = df_oms["response"].apply(contar_palavras)

# Aplica em todos os DataFrames de IA
for df in dfs_ia:
    df["word_count"] = df["response"].apply(contar_palavras)
    df["word_count_diff"] = (df["word_count"] - df_oms["word_count"])

#%% Flesch Reading
# Esta métrica calcula a facilidade de leitura de um texto com base na fórmula Flesch Reading Ease.
# Embora teoricamente a escala varie de 0 a 100, na prática os valores podem ser negativos para textos extremamente complexos.
# Quanto maior o valor, mais fácil é a leitura do texto.

# Interpretação típica da pontuação (aproximada):
# 90–100  → Muito fácil (5º ano)
# 60–70   → Fácil (ensino fundamental)
# 30–50   → Médio a difícil (ensino médio)
# 0–30    → Difícil (nível universitário)
# < 0     → Extremamente difícil

# Interpretação da coluna flesch_diff:
# > 0: resposta da IA está mais fácil de ler que a da OMS
# < 0: resposta da IA está mais difícil
# = 0: mesma facilidade

# Aplica ao df_oms
df_oms["flesch_reading_ease"] = df_oms["response"].apply(textstat.flesch_reading_ease)

# Aplica em cada DataFrame de IA e calcula a diferença
for df in dfs_ia:
    df["flesch_reading_ease"] = df["response"].apply(textstat.flesch_reading_ease)
    df["flesch_diff"] = df["flesch_reading_ease"] - df_oms["flesch_reading_ease"]

#%% Flesch-Kincaid Grade Level
# Esta métrica calcula o Flesch-Kincaid Grade Level, que estima o número de anos de escolaridade
# formal necessários para compreender o texto. Embora a escala normalmente varie entre 0 e 12,
# em textos mais complexos (como respostas longas e técnicas), o valor pode ultrapassar 20, 30 ou mais.

# Interpretação típica (referência teórica):
#  5.0  →  5º ano (fácil)
#  8.0  →  8º ano (intermediário)
# 12.0  →  Ensino médio
# >13.0 →  Nível universitário ou linguagem técnica

# A análise permite comparar a complexidade textual das IAs com a da OMS.

# Interpretação da coluna fk_grade_diff:
# > 0: a resposta da IA exige mais escolaridade que a da OMS (mais difícil)
# < 0: a resposta da IA exige menos escolaridade que a da OMS (mais simples)
# = 0: mesmo nível de escolaridade

# Aplica no df_oms
df_oms["fk_grade_level"] = df_oms["response"].apply(textstat.flesch_kincaid_grade)

# Aplica nos DataFrames das IAs
for df in dfs_ia:
    df["fk_grade_level"] = df["response"].apply(textstat.flesch_kincaid_grade)
    df["fk_grade_diff"] = df["fk_grade_level"] - df_oms["fk_grade_level"]

#%%  Cosine Similarity
# Esta métrica calcula a similaridade semântica entre duas respostas (IA vs OMS) com base no ângulo entre seus vetores TF-IDF.
# Quanto mais próximo de 1, mais semelhantes os textos são em conteúdo.
    # Intervalo da coluna cosine_similarity:
    # 1.0 → textos semanticamente idênticos
    # 0.5 → algum alinhamento temático
    # 0.0 → sem relação semântica

# Prepara o vetor de referência da OMS
oms_respostas = df_oms["response"].fillna("")

# Instancia o vetor TF-IDF
vectorizer = TfidfVectorizer()

# Aplica a cada IA
for df in dfs_ia:
    ia_respostas = df["response"].fillna("")

    # Concatena linha a linha (cada par OMS–IA)
    similaridades = []
    for ia, oms in zip(ia_respostas, oms_respostas):
        tfidf = vectorizer.fit_transform([ia, oms])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        similaridades.append(sim)

    # Adiciona ao DataFrame
    df["cosine_similarity"] = similaridades

#%% Distância de Levenshtein
# Esta métrica calcula a distância de Levenshtein entre os textos da IA e da OMS,
# ou seja, o número mínimo de operações de edição (adição, remoção ou substituição de caracteres) necessárias para transformar um texto no outro.
# Interpretação:
    # 0 → textos idênticos
    # Pequeno valor → textos similares em estrutura
    # Grande valor → textos muito diferentes

# Aplica em cada DataFrame de IA
for df in dfs_ia:
    distancias = []
    for ia_resp, oms_resp in zip(df["response"], df_oms["response"]):
        distancia = Levenshtein.distance(str(ia_resp), str(oms_resp))
        distancias.append(distancia)

    df["levenshtein_dist"] = distancias

#%% Coeficiente de Jaccard    
# Esta métrica calcula o coeficiente de Jaccard entre as respostas da IA e da OMS.
# Ela avalia a similaridade com base na interseção e união dos conjuntos de palavras únicas.
#
# Interpretação da coluna jaccard_similarity:
# 1.0 → textos com vocabulário idêntico
# 0.0 → nenhum vocabulário em comum
# 0.5 Intermediários → parcialmente sobrepostos

def jaccard_similarity(str1, str2):
    set1 = set(str(str1).lower().split())
    set2 = set(str(str2).lower().split())
    if not set1 and not set2:
        return 1.0
    elif not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

# Aplica em cada DataFrame de IA
for df in dfs_ia:
    jaccard_scores = []
    for ia_resp, oms_resp in zip(df["response"], df_oms["response"]):
        score = jaccard_similarity(ia_resp, oms_resp)
        jaccard_scores.append(score)
    df["jaccard_similarity"] = jaccard_scores

#%% Concatenar DFs das IAS
# Concatena diretamente todos os DataFrames das IAs
df_all_responses = pd.concat(dfs_ia, ignore_index=True)

#%%########### ANÁLISE GRÁFICA ###################

#%% Gráfico de barras de Diferença média da quantidade de palavras

plt.figure(figsize=(11, 6))

# Calcula a média por IA
ranking_wc_raw = df_all_responses.groupby("AI")["word_count_diff"].mean()

# Ordena pela distância absoluta em relação a zero
ranking_wc = ranking_wc_raw.reindex((ranking_wc_raw - 0).abs().sort_values().index)

# Cria o gráfico
ax = sns.barplot(x=ranking_wc.values, y=ranking_wc.index, palette="viridis")

# Adiciona rótulos nas extremidades das barras
for i, v in enumerate(ranking_wc.values):
    alinhamento = 'left' if v >= 0 else 'right'
    offset = 0.01 if v >= 0 else -0.01
    ax.text(v + offset, i, f"{v:.1f}", va='center', ha=alinhamento, fontweight='bold')

# Linha de referência
plt.axvline(0, color='purple', linestyle='--')

# Títulos e eixos
plt.title("Diferença média de palavras entre IA e OMS")
plt.xlabel("Palavras a mais (positivas) ou a menos (negativas)")
plt.ylabel("IA")
plt.tight_layout()
plt.show()

# Exportar Gráfico (Opcional)
#plt.savefig("figures/01_diferenca_palavras_ordenado_proximidade_zero.png", dpi=300, bbox_inches="tight")


#%% Gráfico combinado – Flesch Reading Ease (Barplot + Boxplot) ordenado pela proximidade à OMS

# Cálculo da média de Flesch Reading Ease da OMS
media_flesch = df_oms["flesch_reading_ease"].mean()

# Calcular a média por IA
medias_flesch = df_all_responses.groupby("AI")["flesch_reading_ease"].mean()

# Calcular a diferença absoluta em relação à média da OMS
proximidade = (medias_flesch - media_flesch).abs().sort_values()

# Obter a ordem das IAs com base nessa proximidade
ordem_ias = proximidade.index.tolist()

# Criar a figura com dois subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# Barplot (médias)
ranking_flesch = medias_flesch.reindex(ordem_ias)
sns.barplot(x=ranking_flesch.values, y=ranking_flesch.index, palette="viridis", ax=axes[0])
axes[0].axvline(media_flesch, color="red", linestyle="--", label=f"OMS (média: {media_flesch:.2f})")
axes[0].set_title("Flesch Reading Ease – Médias por IA (ordenado por proximidade da OMS)")
axes[0].set_xlabel("")
axes[0].set_ylabel("IA")
axes[0].legend()
for i, v in enumerate(ranking_flesch.values):
    axes[0].text(v + 0.5, i, f"{v:.1f}", va='center', fontweight='bold')

# Boxplot (distribuição por IA)
df_ord = df_all_responses.copy()
df_ord["AI"] = pd.Categorical(df_ord["AI"], categories=ordem_ias, ordered=True)
sns.boxplot(data=df_ord, x="flesch_reading_ease", y="AI", palette="Set3", ax=axes[1])
axes[1].axvline(media_flesch, color="red", linestyle="--", label=f"OMS (média: {media_flesch:.2f})")
axes[1].set_title("Flesch Reading Ease – Distribuição por IA vs Mèdia OMS")
axes[1].set_xlabel("Flesch Reading Ease")
axes[1].set_ylabel("IA")
axes[1].legend()
plt.tight_layout()
plt.show()

# Exportar Gráfico (Opcional)
#plt.savefig("figures/2_flesch_barplot_boxplot_ordenado.png", dpi=300, bbox_inches="tight")

#%% Gráfico combinado – Flesch-Kincaid Grade Level (Barplot + Boxplot) ordenado pela proximidade à OMS

# Cálculo da média da OMS
media_fk = df_oms["fk_grade_level"].mean()

# Cálculo das médias por IA
medias_fk = df_all_responses.groupby("AI")["fk_grade_level"].mean()

# Diferença absoluta em relação à média da OMS
proximidade_fk = (medias_fk - media_fk).abs().sort_values()

# Ordem das IAs pela proximidade
ordem_ias_fk = proximidade_fk.index.tolist()

# Criar a figura
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# Barplot
ranking_fk = medias_fk.reindex(ordem_ias_fk)
sns.barplot(x=ranking_fk.values, y=ranking_fk.index, palette="viridis", ax=axes[0])
axes[0].axvline(media_fk, color="red", linestyle="--", label=f"OMS (média: {media_fk:.2f})")
axes[0].set_title("Flesch-Kincaid Grade Level – Médias por IA (ordenado por proximidade da OMS)")
axes[0].set_xlabel("")
axes[0].set_ylabel("IA")
axes[0].legend()
for i, v in enumerate(ranking_fk.values):
    axes[0].text(v + 0.3, i, f"{v:.1f}", va='center', fontweight='bold')

# Boxplot
df_ord_fk = df_all_responses.copy()
df_ord_fk["AI"] = pd.Categorical(df_ord_fk["AI"], categories=ordem_ias_fk, ordered=True)
sns.boxplot(data=df_ord_fk, x="fk_grade_level", y="AI", palette="Set2", ax=axes[1])
axes[1].axvline(media_fk, color="red", linestyle="--", label=f"OMS (média: {media_fk:.2f})")
axes[1].set_title("Flesch-Kincaid Grade Level – Distribuição por IA vs Média OMS")
axes[1].set_xlabel("Nível Escolar Estimado (Flesch-Kincaid)")
axes[1].set_ylabel("IA")
axes[1].legend()
plt.tight_layout()
plt.show()

# Exportar Gráfico (Opcional)
#plt.savefig("figures/3_fk_grade_barplot_boxplot_ordenado.png", dpi=300, bbox_inches="tight")

#%% Gráfico de barras de cosine similarity por IA
#Ranking: média de cosine similarity por IA
ranking_cosine = df_all_responses.groupby("AI")["cosine_similarity"].mean().sort_values(ascending=False)
ranking_cosine 

# Gráfico de barras com rótulos
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=ranking_cosine.values, y=ranking_cosine.index, palette="viridis")

# Adiciona rótulo em cada barra
for i, v in enumerate(ranking_cosine.values):
    ax.text(v + 0.01, i, f"{v:.2f}", va='center', fontweight='bold')

plt.title("Ranking de IAs por Similaridade de Cosseno com a OMS")
plt.xlabel("Média de Similaridade de Cosseno")
plt.ylabel("IA")
plt.xlim(0, 1.05)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Exportar Gráfico (Opcional)
#plt.savefig("figures/4_cosine_similarity.png", dpi=300, bbox_inches="tight")


#%% Gráfico de barras de Levensthein Distance

plt.figure(figsize=(10, 6))
ranking_lev = df_all_responses.groupby("AI")["levenshtein_dist"].mean().sort_values()
ax = sns.barplot(x=ranking_lev.values, y=ranking_lev.index, palette="viridis")
for i, v in enumerate(ranking_lev.values):
    ax.text(v + 2, i, f"{v:.0f}", va='center', fontweight='bold')
plt.title("Distância de Levenshtein média (IA vs. OMS)")
plt.xlabel("Número médio de edições (caracteres)")
plt.ylabel("IA")
plt.tight_layout()
plt.show()

# Exportar Gráfico (Opcional)
#plt.savefig("figures/5_levensthein_distance.png", dpi=300, bbox_inches="tight")


#%% Gráfico de barras de Jacquard Similarity

plt.figure(figsize=(10, 6))
ranking_jaccard = df_all_responses.groupby("AI")["jaccard_similarity"].mean().sort_values(ascending=False)
ax = sns.barplot(x=ranking_jaccard.values, y=ranking_jaccard.index, palette="viridis")
for i, v in enumerate(ranking_jaccard.values):
    ax.text(v + 0.01, i, f"{v:.2f}", va='center', fontweight='bold')
plt.title("Similaridade de Vocabulário (Jaccard) entre IA e OMS")
plt.xlabel("Coeficiente de Jaccard médio")
plt.ylabel("IA")
plt.xlim(0, 1.05)
plt.tight_layout()
plt.show()

# Exportar Gráfico (Opcional)
#plt.savefig("figures/6_jaccard_coeficient.png", dpi=300, bbox_inches="tight")

#%% Gráfico de Correlação de Pearson das Variáveis

correlation_matrix = df_all_responses[[
    "word_count_diff",
    "fk_grade_level",
    "flesch_reading_ease",
    "cosine_similarity",
    "levenshtein_dist",
    "jaccard_similarity"
]].corr(method="pearson")

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    square=True
)
plt.title("Correlação de Pearson entre Métricas")
plt.tight_layout()
plt.show()

# Exportar Gráfico (Opcional)
#plt.savefig("figures/7_pearson_correlationship.png", dpi=300, bbox_inches="tight")

#%% RANK DE IAs
#Etapas da Metodologia
    #Agrupar por IA e calcular as médias:

    #cosine_similarity → quanto maior, melhor.

    #flesch_reading_ease → quanto maior, mais legível.
    
    #fk_grade_level → quanto menor, mais acessível.
    
    #word_count_diff → ideal é próximo de 0 (valor absoluto menor).
    
    #levenshtein_dist → quanto menor, mais próximo do original.
    
    #jaccard_similarity → quanto maior, mais palavras em comum.
    
    #Normalizar as métricas (escala 0 a 1):
    
    #Métricas positivas: cosine_similarity, flesch_reading_ease, jaccard_similarity
    
    #Métricas negativas (inverter): fk_grade_level, word_count_diff (abs), levenshtein_dist
    
    #Calcular score final como média ponderada (ou simples) das métricas normalizadas.
    
    #Ordenar por score final.

# 1. Agrupamento por IA
agg = df_all_responses.groupby("AI").agg({
   "word_count_diff": lambda x: x.abs().mean(),
   "fk_grade_level": "mean",
   "flesch_reading_ease": "mean",
   "cosine_similarity": "mean",
   "levenshtein_dist": "mean",
   "jaccard_similarity": "mean"
}).reset_index()

# 2. Normalização
scaler = MinMaxScaler()

# Inverter métricas onde menor é melhor
agg["fk_grade_level_inv"] = -agg["fk_grade_level"]
agg["word_count_diff_inv"] = -agg["word_count_diff"]
agg["levenshtein_dist_inv"] = -agg["levenshtein_dist"]

# Apenas as colunas para ranking
rank_columns = [
    "word_count_diff_inv",
    "fk_grade_level_inv",
    "flesch_reading_ease",
    "cosine_similarity",
    "jaccard_similarity",
    "levenshtein_dist_inv"
]

agg_ranked = agg.copy()
agg_ranked[rank_columns] = scaler.fit_transform(agg[rank_columns])

# 3. Score final
agg_ranked["score_final"] = agg_ranked[rank_columns].mean(axis=1)
agg_ranked = agg_ranked.drop(["fk_grade_level", "word_count_diff", "levenshtein_dist"], axis=1)
#ordem de colunas:
    
#Score final no agg
agg["score_final"] = agg_ranked[rank_columns].mean(axis=1)

# Ranking no agg
agg["ranking"] = agg["score_final"].rank(ascending=False).astype(int)

# 4. Ranking
agg_ranked = agg_ranked.sort_values(by="score_final", ascending=False).reset_index(drop=True)
agg_ranked["ranking"] = agg_ranked.index + 1

#ordem das colunas:
agg_ranked = agg_ranked[[ 
 "AI",
 "word_count_diff_inv",
 "fk_grade_level_inv",
 "flesch_reading_ease",
 "cosine_similarity",
 "jaccard_similarity",
 "levenshtein_dist_inv",
 "score_final",
 "ranking"
 ]]


#%% Gráfico de Barras de Score Normalizado de IAs
# Visualizar resultado
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=agg_ranked, x="score_final", y="AI", palette="viridis")
# Adiciona os rótulos de valor à direita das barras
for i, v in enumerate(agg_ranked["score_final"]):
    ax.text(v + 0.005, i, f"{v:.2f}", va='center', fontweight='bold')  
plt.title("Ranking Final das IAs por Desempenho Composto")
plt.xlabel("Score Normalizado (0–1)")
plt.ylabel("IA")
plt.tight_layout()
plt.show()

# Exportar Gráfico (Opcional)
#plt.savefig("figures/8_AI_Rank.png", dpi=300, bbox_inches="tight")


#%% Radar Chart das Top 5 IAs
# Seleciona apenas as 5 melhores IAs
top5 = agg_ranked.head(5).copy()

# Prepara os dados para o radar chart
metrics = [
    "word_count_diff_inv",
    "fk_grade_level_inv",
    "flesch_reading_ease",
    "cosine_similarity",
    "jaccard_similarity",
    "levenshtein_dist_inv"
]

# Normaliza manualmente as métricas para garantir valores de 0 a 1
def min_max_normalization(column):
    return (column - column.min()) / (column.max() - column.min())

radar_data = top5[["AI"] + metrics].copy()
for metric in metrics:
    radar_data[metric] = min_max_normalization(radar_data[metric])

# Reorganiza para o formato de radar
labels = metrics
num_vars = len(labels)

angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # Fechar o gráfico

# Plot
plt.figure(figsize=(8, 8))
for i, row in radar_data.iterrows():
    values = row[metrics].tolist()
    values += values[:1]
    plt.polar(angles, values, label=row["AI"], linewidth=2)
    plt.fill(angles, values, alpha=0.1)

plt.xticks(angles[:-1], labels, fontsize=10)
plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"], color="gray", size=8)
plt.ylim(0, 1)
plt.title("Desempenho por Métrica – Top 5 IAs", size=14)
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plt.show()

# Exportar Gráfico (Opcional)
#plt.savefig("figures/9_radar_chart.png", dpi=300, bbox_inches="tight")

#%%################ ANÁLISE DE IA POR TÓPICO DE PERGUNTAS ####################


#%%Rank de IA por tópico
# Agrupar por IA e tópico e calcular as métricas
agg_topic = df_all_responses.groupby(["AI", "topic"]).agg({
    "word_count_diff": lambda x: x.abs().mean(),
    "fk_grade_level": "mean",
    "flesch_reading_ease": "mean",
    "cosine_similarity": "mean",
    "levenshtein_dist": "mean",
    "jaccard_similarity": "mean"
}).reset_index()

# Inverter métricas onde menor é melhor
agg_topic["fk_grade_level_inv"] = -agg_topic["fk_grade_level"]
agg_topic["word_count_diff_inv"] = -agg_topic["word_count_diff"]
agg_topic["levenshtein_dist_inv"] = -agg_topic["levenshtein_dist"]

# Apenas as colunas para ranking
rank_columns = [
    "word_count_diff_inv",
    "fk_grade_level_inv",
    "flesch_reading_ease",
    "cosine_similarity",
    "jaccard_similarity",
    "levenshtein_dist_inv"
]

# Normalização min-max por tópico

agg_topic_ranked = agg_topic.copy()
for topic in agg_topic["topic"].unique():
    scaler = MinMaxScaler()
    mask = agg_topic["topic"] == topic
    agg_topic_ranked.loc[mask, rank_columns] = scaler.fit_transform(agg_topic.loc[mask, rank_columns])

# Score final por tópico
agg_topic_ranked["score_topic"] = agg_topic_ranked[rank_columns].mean(axis=1)

# Ranking por tópico
agg_topic_ranked["ranking_topic"] = agg_topic_ranked.groupby("topic")["score_topic"].rank(ascending=False).astype(int)
agg_topic_ranked = agg_topic_ranked.drop(["fk_grade_level", "word_count_diff", "levenshtein_dist"], axis=1)


#Incluir score e rank diretamente no agg_topic original
agg_topic["score_topic"] = agg_topic_ranked["score_topic"]
agg_topic["ranking_topic"] = agg_topic_ranked["ranking_topic"]

#%% Gráfico de Visualização de ranking final por tópico
# Cria grid 2x2
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Lista de tópicos únicos
topics = agg_topic_ranked["topic"].unique()

# Gera gráfico por tópico
for i, topic in enumerate(topics):
    df_topic = agg_topic_ranked[agg_topic_ranked["topic"] == topic].sort_values("score_topic", ascending=False)
    
    ax = axes[i]
    sns.barplot(data=df_topic, x="score_topic", y="AI", ax=ax, palette="viridis")
    
    for j, score in enumerate(df_topic["score_topic"]):
        ax.text(score + 0.01, j, f"{score:.2f}", va='center', fontweight='bold')
    
    ax.set_title(f"Tópico: {topic}", fontsize=14)
    ax.set_xlabel("Score Normalizado")
    ax.set_ylabel("IA")
    ax.set_xlim(0, 1.05)

# Título geral
plt.suptitle("Ranking das IAs por Tópico (4 Quadrantes)", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Exportar Gráfico (Opcional)
#plt.savefig("figures/10_IA_Topic_Rank.png", dpi=300, bbox_inches="tight")


#%%######################### CLUSTER e ANACOR ##########################

#Cluster no df_agg_topic para obter grupos de respostas de acordo com as metricas obtidas

#%% Preparação do DF
# Separando somente as variáveis quantitativas do banco de dados df_agg_topic
df_agg_topic_quanti = agg_topic.drop(['AI', 'topic', 'score_topic', 'ranking_topic'],axis=1)

# Estatísticas descritivas das variáveis
df_agg_topic_quanti.describe()

#%% Realizando a padronização por meio do Z-Score
# As variáveis estão em unidades de medidas distintas e precisam ser padronizadas
df_agg_topic_pad = df_agg_topic_quanti.apply(zscore, ddof=1)

#%% Identificação da quantidade de clusters (Método Elbow)

elbow = []
K = range(1,11) # ponto de parada pode ser parametrizado manualmente
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=42).fit(df_agg_topic_pad)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,11)) # ajustar range
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow', fontsize=16)
plt.show()

# Exportar Gráfico (Opcional)
#plt.savefig("figures/11_clusters_elbow", dpi=300, bbox_inches="tight")

#%% Identificação da quantidade de clusters (Método da Silhueta) busca-se o ponto máximo

silhueta = []
I = range(2,11) # ponto de parada pode ser parametrizado manualmente
for i in I: 
    kmeansSil = KMeans(n_clusters=i, init='random', random_state=100).fit(df_agg_topic_pad)
    silhueta.append(silhouette_score(df_agg_topic_pad, kmeansSil.labels_))

plt.figure(figsize=(16,8))
plt.plot(range(2, 11), silhueta, color = 'purple', marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Silhueta Média', fontsize=16)
plt.title('Método da Silhueta', fontsize=16)
plt.axvline(x = silhueta.index(max(silhueta))+2, linestyle = 'dotted', color = 'red')
plt.show()

# Exportar Gráfico (Opcional)
#plt.savefig("figures/12_clusters_silhouette.png", dpi=300, bbox_inches="tight") 

 #%% Cluster K-means

# Considerar 3 clusters, dadas as evidências anteriores
kmeans_final = KMeans(n_clusters = 3, init = 'random', random_state=100).fit(df_agg_topic_pad)

# Gerando a variável para identificarmos os clusters gerados
kmeans_clusters = kmeans_final.labels_
agg_topic['Cluster'] = kmeans_clusters
df_agg_topic_pad['Cluster'] = kmeans_clusters
agg_topic_ranked['Cluster'] = kmeans_clusters

#transformação da variável cluster em category para ser utilizada na Anacor
agg_topic['Cluster'] = agg_topic['Cluster'].astype('category')
agg_topic_ranked['Cluster'] = agg_topic_ranked['Cluster'].astype('category')
df_agg_topic_pad['Cluster'] = df_agg_topic_pad['Cluster'].astype('category')

#ordem de colunas:
agg_topic_ranked = agg_topic_ranked [[
   "AI",
  "topic",
  "word_count_diff_inv",
  "fk_grade_level_inv",
  "flesch_reading_ease",
  "cosine_similarity",
  "jaccard_similarity",
  "levenshtein_dist_inv",
  "score_topic",
  "ranking_topic",
  "Cluster"]]

#%% Teste de ANOVA para as variáveis do cluster

# Objetivo:
# Verificar se os clusters gerados são estatisticamente diferentes em relação a cada uma das métricas avaliadas.
# A ANOVA (Análise de Variância) testa se a média de uma variável contínua difere entre dois ou mais grupos (neste caso, clusters).

# Interpretação:
# - p-valor < 0.05 → há evidência estatística de que pelo menos um cluster é diferente dos outros.
# - F alto → maior separação entre os clusters para aquela métrica.
# - Eta² parcial (np2) → tamanho do efeito (acima de 0.14 já é considerado forte).

# Lista das métricas a serem testadas
metricas = [
    "word_count_diff",
    "fk_grade_level",
    "flesch_reading_ease",
    "cosine_similarity",
    "jaccard_similarity",
    "levenshtein_dist"
]

# Lista para armazenar os resultados
anova_results = []

# Loop para aplicar ANOVA em cada métrica
for metrica in metricas:
    resultado = pg.anova(
        dv=metrica,               # variável dependente (métrica)
        between='Cluster',        # variável independente (grupos = cluster)
        data=agg_topic,           # DataFrame com dados
        detailed=True             # incluir F, p, eta² parcial etc.
    )
    
    # Impressão do resultado individual
    print(f"\nANOVA para: {metrica}\n")
    print(resultado)
    
    # Armazenamento
    row = {
        'Métrica': metrica,
        'F': resultado.loc[0, 'F'],
        'p-valor': resultado.loc[0, 'p-unc'],
        'Eta² parcial': resultado.loc[0, 'np2']
    }
    anova_results.append(row)

# Criação do DataFrame com os resultados
df_anova = pd.DataFrame(anova_results)
print("\nResumo dos Resultados da ANOVA:")
print(df_anova)

#%% Boxplots das métricas por cluster
plt.figure(figsize=(16, 12))
for i, metrica in enumerate(metricas):
    plt.subplot(3, 2, i + 1)
    sns.boxplot(data=agg_topic, x="Cluster", y=metrica, palette="Set2")
    plt.title(f"{metrica} por Cluster")
    plt.xlabel("Cluster")
    plt.ylabel(metrica)

plt.suptitle("Distribuição das Métricas por Cluster", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Exportar Gráfico (Opcional)
#plt.savefig("figures/13_clusters_metrics.png", dpi=300, bbox_inches="tight") 

#%% Boxplots do score_topic por Cluster
# Objetivo: ver se há diferenças sistemáticas no desempenho médio entre os clusters.
# Interpretação: se os boxplots têm níveis diferentes de score, os clusters têm desempenho distinto.
plt.figure(figsize=(8, 6))
sns.boxplot(data=agg_topic, x="Cluster", y="score_topic", palette="Set2")
plt.title("Distribuição do Score por Cluster")
plt.xlabel("Cluster")
plt.ylabel("Score Médio por Tópico")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# Exportar Gráfico (Opcional)
#plt.savefig("figures/14_cluster_topic.png", dpi=300, bbox_inches="tight") 


# Média de score por cluster
media_score_cluster = agg_topic.groupby("Cluster")["score_topic"].mean().reset_index()
media_score_cluster.columns = ["Cluster", "score_medio"]
media_score_cluster["ranking_cluster"] = media_score_cluster["score_medio"].rank(ascending=False).astype(int)
print(media_score_cluster)

# Correlação entre score médio e ranking dos clusters
correlacao = media_score_cluster["score_medio"].corr(media_score_cluster["ranking_cluster"], method="spearman")
print(f"Coeficiente de Correlação de Spearman: {correlacao:.2f}")

#Correlação alta (|ρ| ≥ 0.7): seus clusters refletem bem o desempenho das IAs por tópico.
#Boxplots bem separados: reforçam que os clusters são internamente coesos e externamente distintos.

#%% Características médias dos clusters nas métricas avaliadas

# Objetivo:
# Comparar os clusters formados com base na média das principais métricas avaliadas (semelhança, legibilidade, etc.)
# Isso ajuda a entender o "perfil" típico de cada cluster.

# Calcula as médias por cluster
cluster_summary = agg_topic.groupby("Cluster")[[
   "word_count_diff",
   "fk_grade_level",
   "flesch_reading_ease",
   "cosine_similarity",
   "jaccard_similarity",
   "levenshtein_dist"
]].mean().round(2)

#%% ANACOR MÚLTIPLA

#%% Inicio

# Realizar uma ANACOR MULTIPLA (ACM) nas variáveis qualitativas (incluir os clusters!)

# Separando somente as variáveis categóricas siguinificativas para ANACOR no banco de dados 
# As variáveis question e response não fazem sentido na Análise de Correspondência) porque:
#São variáveis textuais livres (não categóricas simples)
df_agg_topic_quali = agg_topic[['AI', 'topic', 'Cluster']]

#%% Teste Qui-Quadrado entre variáveis qualitativas (usando df_all_responses_quali)
# Objetivo:
# Verificar se há associação estatisticamente significativa entre a variável 'topic'
# e outras variáveis qualitativas como IA e Cluster.


# Lista de variáveis categóricas a serem testadas contra 'topic'
variaveis_quali = ['AI', 'Cluster']  # Ajuste conforme as colunas disponíveis em df_all_responses_quali

# Loop pelos testes de associação
print("Teste de associação com a variável 'topic':\n")
for var in variaveis_quali:
    tabela = pd.crosstab(df_agg_topic_quali["topic"], df_agg_topic_quali[var])
    chi2, p, dof, expected = chi2_contingency(tabela)
    print(f"- {var}: p-valor = {round(p, 4)}")

# Interpretação:
# p < 0.05 → Existe associação estatística entre 'topic' e a variável categórica analisada.
# p >= 0.05 → Não há evidência de associação.

# o resultado indica: "Certos padrões de escrita, legibilidade, similaridade, vocabulário ou estrutura textual estão mais associados a tópicos específicos."

#%% Elaborando a análise de correspondência múltipla com duas dimensões!

# Criando coordenadas para 3 dimensões (a seguir, verifica-se a viabilidade)
mca = prince.MCA(n_components=2).fit(df_agg_topic_quali)

# Executa a MCA
mca = prince.MCA(n_components=2, random_state=42)
mca = mca.fit(df_agg_topic_quali)

#Analisando os resultados

# Análise dos autovalores
tabela_autovalores = mca.eigenvalues_summary
print(tabela_autovalores)

# Inércia total da análise
print(mca.total_inertia_)

# Plotar apenas dimensões com inércia parcial superior à inércia total média
quant_dim = mca.J_ - mca.K_
print(mca.total_inertia_/quant_dim)


# Inércia explicada por dimensão
eigenvalues = mca.eigenvalues_

# Coordenadas das categorias
coord_col = mca.column_coordinates(df_agg_topic_quali).reset_index()
coord_col.columns = ['Categoria', 'Dim1', 'Dim2']

# Plot do gráfico (somente categorias)
# Classificar tipo de categoria: AI, topic, Cluster
def classificar_categoria(cat):
    if cat.startswith("AI_"):
        return "AI"
    elif cat.startswith("topic_"):
        return "Tópico"
    elif cat.startswith("Cluster_"):
        return "Cluster"
    else:
        return "Outro"

# Aplicar classificação
coord_col["tipo"] = coord_col["Categoria"].apply(classificar_categoria)

# Definir cores por tipo
cores = {"AI": "blue", "Tópico": "orange", "Cluster": "purple"}

# Plot customizado
plt.figure(figsize=(10, 8))
for tipo, cor in cores.items():
    dados = coord_col[coord_col["tipo"] == tipo]
    plt.scatter(dados["Dim1"], dados["Dim2"], label=tipo, color=cor, s=70)
    for _, row in dados.iterrows():
        plt.text(row["Dim1"] + 0.03, row["Dim2"] - 0.02, row["Categoria"], fontsize=9)

# Linhas e legendas
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.title("Mapa Perceptual – MCA (AI, Tópico e Cluster)", fontsize=14, weight="bold")
plt.xlabel(f"Dim 1: {round(eigenvalues[0] * 100, 2)}% da inércia", fontsize=10)
plt.ylabel(f"Dim 2: {round(eigenvalues[1] * 100, 2)}% da inércia", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(title="Tipo de Categoria")
plt.tight_layout()
plt.show()

# Exportar Gráfico (Opcional)
#plt.savefig("figures/15_anacor_2_dimensions.png", dpi=300, bbox_inches="tight") 

#%% Elaborando a análise de correspondência múltipla com 3 dimnesões!

# Criando coordenadas para 3 dimensões (a seguir, verifica-se a viabilidade)
mca = prince.MCA(n_components=3).fit(df_agg_topic_quali)

#Analisando os resultados

# Análise dos autovalores
tabela_autovalores = mca.eigenvalues_summary
print(tabela_autovalores)

# Inércia total da análise
print(mca.total_inertia_)

# Plotar apenas dimensões com inércia parcial superior à inércia total média
quant_dim = mca.J_ - mca.K_
print(mca.total_inertia_/quant_dim)

# Obtendo as coordenadas-padrão das categorias das variáveis
coord_padrao = mca.column_coordinates(df_agg_topic_quali)/np.sqrt(mca.eigenvalues_)
print(coord_padrao)

#Plotando o mapa perceptual (coordenadas-padrão)
# Primeiro passo: gerar um DataFrame detalhado
chart = coord_padrao.reset_index()
var_chart = pd.Series(chart['index'].str.split('_', expand=True).iloc[:,0])

nome_categ=[]
for col in df_agg_topic_quali:
    nome_categ.append(df_agg_topic_quali[col].sort_values(ascending=True).unique())
    categorias = pd.DataFrame(nome_categ).stack().reset_index()

chart_df_mca = pd.DataFrame({'categoria': chart['index'],
                             'obs_x': chart[0],
                             'obs_y': chart[1],
                             'obs_z': chart[2],
                             'variavel': var_chart,
                             'categoria_id': categorias[0]})

# Segundo passo: gerar o gráfico de pontos
fig = px.scatter_3d(chart_df_mca, 
                    x='obs_x', 
                    y='obs_y', 
                    z='obs_z',
                    color='variavel',
                    text=chart_df_mca.categoria_id)

fig.write_html("results/16_grafico_mca_3d.html")
webbrowser.open("results/16_grafico_mca_3d.html")
fig.show()

#%% Exportação de principais arquivos (Opcional)

# df_all_responses.to_excel("results/df_all_responses.xlsx", index=False)
# df_oms.to_excel("results/df_oms_metrics.xlsx", index=False)
# agg_ranked.to_excel("results/df_agg_ranked.xlsx", index=False)
# agg_topic_ranked.to_excel("results/df_agg_topic_cluster.xlsx", index=False)
