from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def prepareDataSet():
    print("Carregando dataset Hepatitis...")

    # OpenML ID 55: Hepatitis
    data = fetch_openml(data_id = 55, as_frame = True, parser = "auto")
    df = data.frame
    
    # Separar features e alvo (Class é a coluna alvo)
    x = df.drop(columns="Class")
    y_raw = df["Class"]

    # Binarizar o target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Identificar colunas
    numeric_features = x.select_dtypes(include=np.number).columns.tolist()
    categorical_features = x.select_dtypes(include=["category", "object"]).columns.tolist()

    print(f"Features numéricas: {len(numeric_features)}")
    print(f"Features categóricas: {len(categorical_features)}")
    print(f"Total de features: {len(x.columns)}")
    print(f"Amostras: {len(x)}")

    return x, y, numeric_features, categorical_features

def runExperiments(x, y, numeric_features, categorical_features):
    results = []    

    # Imputação
    imputers_list = {
        "Mediana": SimpleImputer(strategy = "median"),
        "KNNImputer": KNNImputer(n_neighbors = 5)
    }

    # Scaling
    scalers_list = {
        "Standard": StandardScaler(),
        "MinMax": MinMaxScaler()
    }

    # Balanceamento
    balancers_list = {
        "Nenhum": None,  # Passthrough
        "ROS": RandomOverSampler(random_state = 42)
    }

    # Seleção de Atributos
    selectors_list = {
        "Nenhuma": "passthrough", # Mantém todas as colunas
        "SelectKBest": SelectKBest(score_func = f_classif, k = 10) # Seleciona as 10 melhores
    }

    experiment_id = 1
    total_experiments = 16

    print(f"\nIniciando os {total_experiments} experimentos com Cross-Validation (k=5)...\n")

    # Loop das 16 combinações
    for imp_name, imputer_algo in imputers_list.items():
        for scl_name, scaler_algo in scalers_list.items():
            for bal_name, balancer_algo in balancers_list.items():
                for sel_name, selector_algo in selectors_list.items():
                    
                    print(f"Exp {experiment_id}/{total_experiments}: {imp_name} + {scl_name} + {bal_name} + {sel_name}")

                    # Pré-processamento das colunas
                    # Numéricas com Imputer Variável + Scaler Variável
                    num_transformer = ImbPipeline(steps=[('imputer', imputer_algo), ('scaler', scaler_algo)])

                    # Categóricas: Imputer da Moda + OneHot
                    cat_transformer = ImbPipeline(steps=[('imputer', SimpleImputer(strategy = 'most_frequent')), ('encoder', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False))])

                    preprocessor = ColumnTransformer(transformers = [('num', num_transformer, numeric_features), ('cat', cat_transformer, categorical_features)])

                    # Pipeline Principal
                    steps = [('preprocessor', preprocessor), ('balancer', balancer_algo), ('selector', selector_algo), ('classifier', KNeighborsClassifier(n_neighbors  = 5))]
                    
                    # Remover passos que são None
                    steps = [s for s in steps if s[1] is not None]
                    
                    model_pipeline = ImbPipeline(steps = steps)

                    # Validação Cruzada
                    cv = StratifiedKFold(n_splits = 5, shuffle=True, random_state = 42)
                    
                    # Executa o CV e coleta métricas
                    cv_results = cross_validate(model_pipeline, x, y, cv = cv, scoring = 'f1_weighted', n_jobs = -1)
                    
                    mean_f1 = cv_results['test_score'].mean()
                    std_f1 = cv_results['test_score'].std()

                    # Armazenar resultados
                    results.append({
                        "ID": experiment_id,
                        "Imputação": imp_name,
                        "Scaling": scl_name,
                        "Balanceamento": bal_name,
                        "Seleção": sel_name,
                        "F1-Score Médio": mean_f1,
                        "Desvio Padrão": std_f1
                    })

                    experiment_id += 1

    return pd.DataFrame(results)

def analyzeAndPlot(df_results):
    print("\nTabela Consolidada dos Resultados\n")

    # Ordenar pelo melhor F1-Score
    df_sorted = df_results.sort_values(by = "F1-Score Médio", ascending = False)
    print(df_sorted.to_markdown(index = False, floatfmt = ".4f"))

    # Plotando os resultados
    plt.figure(figsize = (14, 8))
    
    # Criar um rótulo combinado para o eixo X
    labels = df_sorted.apply(lambda row: f"{row['Imputação']}\n{row['Scaling']}\n{row['Balanceamento']}\n{row['Seleção']}", axis = 1)
    
    plt.bar(labels, df_sorted["F1-Score Médio"], yerr = df_sorted["Desvio Padrão"], capsize = 5, color = 'skyblue', edgecolor = 'black')
    
    plt.title("Comparação dos 16 Pipelines (F1-Score Weighted)", fontsize = 16)
    plt.ylabel("F1-Score Médio (5-Fold CV)", fontsize = 12)
    plt.xlabel("Combinação de Técnicas", fontsize = 12)
    plt.xticks(rotation = 90, fontsize = 9)
    plt.ylim(0.0, 1.0)
    plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)
    
    plt.tight_layout()
    plt.show()

    # Melhor resultado
    best = df_sorted.iloc[0]
    print(f"\nMelhor Combinação: Exp {best['ID']}")
    print(f"Configuração: {best['Imputação']} | {best['Scaling']} | {best['Balanceamento']} | {best['Seleção']}")
    print(f"F1-Score: {best['F1-Score Médio']:.4f}")

x, y, num_feats, cat_feats = prepareDataSet()

results_df = runExperiments(x, y, num_feats, cat_feats)
analyzeAndPlot(results_df)