import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, roc_auc_score

# Cargar datos por ciclo
train_1 = pd.read_csv("data\datos_completos_2007_2008.csv")
train_2 = pd.read_csv("data\datos_completos_2009_2010.csv")
test = pd.read_csv("data\datos_completos_2011_2012.csv")

# Unir ciclos de entrenamiento
train = pd.concat([train_1, train_2], ignore_index=True)

# Definir variables deseadas
features = ["RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR",
            "BPXSY1", "BPXDI1", "BMXBMI", "BMXWAIST",
            "LBDHDL", "LBXTC", "LBXTR", "LBXGH", "LBXCRP"]

# Filtrar columnas disponibles en cada set
features_train = [col for col in features if col in train.columns]
features_test = [col for col in features if col in test.columns]

X_train = train[features_train]
X_test = test[features_test]

# Definir target combinado: glucosa alta o hipertensión
train["riesgo_cardiometabolico"] = (
    (train["LBXGLU"] >= 100) | (train["BPXSY1"] >= 130) | (train["BPXDI1"] >= 80)
).astype(int)

test["riesgo_cardiometabolico"] = (
    (test["LBXGLU"] >= 100) | (test["BPXSY1"] >= 130) | (test["BPXDI1"] >= 80)
).astype(int)

y_train = train["riesgo_cardiometabolico"]
y_test = test["riesgo_cardiometabolico"]

# Entrenar modelo de regresión logística
modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
modelo.fit(X_train, y_train)

# Evaluar rendimiento
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# Mostrar coeficientes
coefs = modelo.named_steps["logisticregression"].coef_[0]
for nombre, valor in zip(features_train, coefs):
    print(f"{nombre}: {valor:.4f}")
