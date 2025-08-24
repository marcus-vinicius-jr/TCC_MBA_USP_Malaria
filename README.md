# TCC – MBA USP/Esalq (Data Science & Analytics)
## Avaliação de Modelos Generativos em Saúde: estudo de caso sobre Malária

Repositório com **dados, código e produtos analíticos** do TCC:
**“Avaliação de Modelos de Inteligência Artificial Generativa em Comparação com Informações da OMS sobre Malária”**.

- Autor: **Marcus Vinicius Freire Junior**  
- Curso: **MBA em Data Science and Analytics – USP/Esalq**  
- Ano: **2025**

---

## 📦 Estrutura do repositório

├── data/ # Bases de entrada (IA e OMS)  
│ ├── df_claude.xlsx  
│ ├── df_copilot.xlsx  
│ ├── df_deepseek.xlsx  
│ ├── df_gemini.xlsx  
│ ├── df_gpt.xlsx  
│ ├── df_gpt_escholar.xlsx  
│ ├── df_gpt_vision.xlsx  
│ ├── df_meta_llama.xlsx  
│ ├── df_perplexity.xlsx  
│ ├── df_reka.xlsx  
│ └── df_oms.xlsx  
│  
├── code/  
│ ├── TCC_MBA_DSA_USP_MALARIA.py # Script principal (Python)  
│ └── requirements.txt  
│  
├── figures/ # Imagens estáticas exportadas (PNG)  
│ └── (geradas pelo script)  
│  
├── results/ # Saídas tabulares/HTML (MCA 3D etc.)  
│ └── (geradas pelo script)  

## ⚙️ Ambiente e dependências

### Instalação 
```bash
pip install -r requirements.txt

Principais bibliotecas:
- pandas, numpy, scikit-learn, scipy, statsmodels, pingouin
- textstat, python-Levenshtein
- matplotlib, seaborn, plotly, prince
- (opcional para exportar PNG de gráficos plotly) kaleido

---

## ▶️ Como executar

1. Certifique-se de que as bases estão em `data/`.
2. Execute o script principal:
 python code/TCC_MBA_DSA_USP_MALARIA.py
3. Saídas geradas automaticamente:
- Figuras (PNG) em `figures/`
- Tabelas (XLSX/CSV) em `results/`
- Gráfico interativo MCA 3D em `results/16_grafico_mca_3d.html`

---

## 🧪 Metodologia

- **Legibilidade**: Flesch Reading Ease e Flesch–Kincaid Grade Level  
- **Similaridade textual**: Similaridade do Cosseno, Distância de Levenshtein, Coeficiente de Jaccard  
- **Clusterização**: K-means com padronização por Z-score; seleção de k via Elbow e Silhueta  
- **MCA (Análise de Correspondência Múltipla)**: associação entre IA, tópicos e clusters  
- **Validação**: ANOVA (diferenças de médias) e Qui-quadrado (associação entre categorias)  

---

## 📊 Produtos gerados

- Rankings por IA e por tópico  
- Matriz de correlação (Pearson)  
- Determinação do número de clusters (Elbow / Silhueta)  
- Boxplots e gráficos comparativos  
- Mapa perceptual – MCA 2D e 3D  
- Tabelas agregadas (`results/df_agg_ranked.xlsx`, `results/df_agg_topic_cluster.xlsx`) 

---

##📑 Citação sugerida

FREIRE JUNIOR, Marcus Vinicius. Avaliação de Modelos de Inteligência Artificial Generativa em
Comparação com Informações da OMS sobre Malária. Trabalho de Conclusão de Curso (MBA em Data
Science and Analytics) – Universidade de São Paulo, Escola Superior de Agricultura “Luiz de Queiroz”,
2025.
