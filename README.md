# BMW Global Sales Analysis (2010–2024)

Analisi completa delle vendite BMW nel mondo (2010–2024) con Python e Power BI.  
Il progetto include:
- Pulizia e aggregazione dei dati (`bmw_sales_clean.csv`)
- Analisi statistica (correlazioni, ANOVA, regressione lineare e log-lineare)
- Dashboard Power BI interattiva (trend YoY, regioni, fuel type)

---

## 📊 Fasi del progetto

1. **Data Cleaning & Exploration**
   - Analisi esplorativa e descrittiva in Python.
   - Visualizzazioni (heatmap, boxplot).

2. **Statistical Analysis**
   - Test ANOVA: differenze di prezzo non significative (*p* = 0.53).
   - Correlazioni e trend aggregati per anno, regione, modello.

3. **Regression Analysis**
   - Regressione OLS con effetti fissi di regione e anno.
   - Elasticità prezzo-vendite: **−0.0465** (*p* = 0.82) → domanda inelastica.
   - Fuel type ibrido: **+3.4%** (*p* = 0.02).
   - Mercato asiatico: crescita **+3%** (*p* = 0.06).

4. **Power BI Dashboard**
   - Pagina Executive: KPI e trend globali.
   - Market Breakdown: Top regioni, modelli e fuel mix.

---

## 🧠 Conclusioni

- La **domanda BMW è inelastica** al prezzo (segmento premium).
- Gli **ibridi** sono il principale motore di crescita.
- L’**Asia** è la regione trainante delle vendite globali.
- Il **2022** segna un rimbalzo significativo post-pandemia (+6%).

---

## ⚙️ Tecnologie

- **Python** (pandas, seaborn, statsmodels)
- **Power BI**
- **GitHub**

---
## 👤 Autore
**Massimiliano Piccolo**  
Junior Data Analyst & Data Scientist in formazione