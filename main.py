import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import glob

print("Aguarde q leva um tempo kk...")
print("Tempo que levou para mim: 1m 17s")

caminho_arquivos = "./datasets/"

# Encontrar e carregar todos os arquivos CSV de treino em um único DataFrame
arquivos_treino = glob.glob(caminho_arquivos + "lbp-train-fold_*.csv")
train_df = pd.concat([pd.read_csv(arquivo) for arquivo in arquivos_treino], ignore_index=True)

# Carregar os dados de teste
test_df = pd.read_csv(caminho_arquivos + "lbp-test.csv")

# Separar features e rótulo nos conjuntos de treino e teste
X_train = train_df.drop(columns=['class'])
y_train = train_df['class']
X_test = test_df.drop(columns=['class'])
y_test = test_df['class']

# Seleção de características usando RandomForest
selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="mean")
selector.fit(X_train, y_train)
selected_features = X_train.columns[selector.get_support()]

# Aplicar seleção de características
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# RandomOverSampler para balancear as classes no conjunto de treino
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)

# Inicializar e configurar os modelos para Ensemble
rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)
svc_model = SVC(probability=True, random_state=42)

# Combinar os modelos com VotingClassifier
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('svc', svc_model)
    ], voting='soft'
)

# Hiperparâmetros para ajuste de modelo
param_grid = {
    'rf__n_estimators': [100, 200],
    'xgb__max_depth': [3, 5],
    'svc__C': [1, 10]
}

# Usar GridSearchCV para ajuste dos hiperparâmetros
grid_search = GridSearchCV(ensemble_model, param_grid, scoring='accuracy', cv=3)
grid_search.fit(X_train_balanced, y_train_balanced)

# Melhor modelo encontrado
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_selected)
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))
print("Acurácia no conjunto de teste:", accuracy_score(y_test, y_pred))
