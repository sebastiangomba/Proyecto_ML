# ==============================================================
# Machine Learning I - Maestría en Analítica Aplicada
# Universidad de la Sabana
# Prof: Hugo Franco
# Grupo: Sebastián Gómez (0000261948), Leonardo Montoya (0000286207) y Andrea Urdaneta (0000257784)
# ============================================================
# Proyecto Final: Pipeline completo con Prefect en relación con un dataset relativo a las pruebas SABER 11 y SABER PRO
# Clasificación (target=quartile)
# Regresión (target=total_score)
# Reporte final con mejor modelo
# =============================================================

"""
Requisitos:
    pip install pandas numpy scikit-learn imbalanced-learn xgboost joblib prefect openpyxl
    ajustar N_ITER, CV_FOLDS, N_JOBS según recursos
"""
import os
import warnings
warnings.filterwarnings("ignore")

# Fuerza backend no interactivo para evitar NSWindow en macOS/hilos
import matplotlib
matplotlib.use("Agg")  # <- clave

# ahora sí importa pyplot
import matplotlib.pyplot as plt

# -------------------------
# PARÁMETROS GLOBALES
# -------------------------
INPUT_PATH = "//Users/sebastiangomba/Documents/Universidad/Machine Learning I/Proyecto/Dataset/data_academic_performance.xlsx"
OUTPUT_DIR = "//Users/sebastiangomba/Documents/Universidad/Machine Learning I/Proyecto/Resultados"     # directorio para guardar modelos/plots/imágenes
N_ITER = 2
CV_FOLDS = 3
N_JOBS = -1
RANDOM_STATE = 42
USE_XGBOOST = False

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# IMPORTS
# -------------------------
import pandas as pd
import numpy as np
from joblib import dump
from prefect import flow, task

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report,
    mean_squared_error
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

if USE_XGBOOST:
    from xgboost import XGBClassifier, XGBRegressor

# ============================================================
# TAREAS PREFECT
# ============================================================

@task
def load_data(path):
    print("Cargando datos...")
    df = pd.read_excel(path)
    print("Shape:", df.shape)
    return df


@task
def preprocess_data(df):
    print("Preparando features y columnas...")

    quartile_col = next((c for c in df.columns if "quart" in c.lower()), None)
    if not quartile_col:
        raise ValueError("No se encontró columna 'quartile' en el dataset.")

    score_cols = ['MAT_S11', 'CR_S11', 'CC_S11', 'BIO_S11', 'ENG_S11']
    df['total_score'] = df[score_cols].sum(axis=1)

    exclude = [quartile_col, 'total_score']
    for c in df.columns:
        if str(c).lower() in ['id', 'index']:
            exclude.append(c)

    feature_cols = [c for c in df.columns if c not in exclude]
    num_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_features = [c for c in feature_cols if c not in num_features]

    df[cat_features] = df[cat_features].astype(str)

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ])

    return df, feature_cols, quartile_col, preprocessor


@task
def train_classification(df, feature_cols, quartile_col, preprocessor):
    print("\nEntrenando modelos de clasificación...")

    X = df[feature_cols]
    y = df[quartile_col]

    if y.min() > 0:
        y = y - y.min()
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    classifiers = {
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True),
        "Logistic": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE)
    }

    param_distributions = {
        "KNN": {"model__n_neighbors": [3, 5]},
        "SVM": {"model__C": [0.1, 1], "model__kernel": ['rbf']},
        "Logistic": {"model__C": [0.1, 1]},
        "RandomForest": {"model__n_estimators": [50, 100], "model__max_depth": [None, 10]},
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, model in classifiers.items():
        print(f"\n== Entrenando {name} ==")
        pipe = ImbPipeline([
            ("preproc", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model)
        ])
        rs = RandomizedSearchCV(
            pipe, param_distributions[name], n_iter=N_ITER,
            scoring="balanced_accuracy", cv=cv, n_jobs=N_JOBS, random_state=RANDOM_STATE
        )
        rs.fit(X_train, y_train)
        y_pred = rs.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        print(f"Test Accuracy: {acc:.3f} | Balanced Accuracy: {bal_acc:.3f}")
        print("\nMatriz de confusión:")
        print(cm)
        print("\nReporte de clasificación:")
        print(cr)

        results.append((name, acc, bal_acc, rs.best_estimator_, rs.best_params_))
        dump(rs.best_estimator_, os.path.join(OUTPUT_DIR, f"{name}_classifier.joblib"))

    df_results = pd.DataFrame(results, columns=["Modelo", "Accuracy", "BalancedAcc", "BestEstimator", "BestParams"])
    df_results.to_csv(os.path.join(OUTPUT_DIR, "comparativa_clasificacion.csv"), index=False)
    return df_results


@task
def train_regression(df, feature_cols, preprocessor):
    print("\nEntrenando modelos de regresión...")

    X = df[feature_cols]
    y = df['total_score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    regressors = {
        "KNN": KNeighborsRegressor(),
        "SVR": SVR()
    }

    param_distributions = {
        "KNN": {"model__n_neighbors": [3, 5]},
        "SVR": {"model__C": [0.1, 1]},
    }

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, model in regressors.items():
        print(f"== Entrenando (reg) {name} ==")
        pipe = Pipeline([("preproc", preprocessor), ("model", model)])
        rs = RandomizedSearchCV(pipe, param_distributions[name], n_iter=N_ITER,
                                scoring="neg_root_mean_squared_error", cv=cv, n_jobs=N_JOBS)
        rs.fit(X_train, y_train)
        y_pred = rs.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"{name} RMSE: {rmse:.3f}")

        results.append((name, rmse))
        dump(rs.best_estimator_, os.path.join(OUTPUT_DIR, f"{name}_regressor.joblib"))

    df_results = pd.DataFrame(results, columns=["Modelo", "RMSE"])
    df_results.to_csv(os.path.join(OUTPUT_DIR, "comparativa_regresion.csv"), index=False)
    return df_results


@task
def best_model_report(df_results):
    print("\nDeterminando mejor modelo de clasificación...")
    best_row = df_results.sort_values(by="BalancedAcc", ascending=False).iloc[0]
    best_model = best_row['Modelo']
    best_params = best_row['BestParams']

    print(f"\nMejor modelo: {best_model}")
    print(f"Accuracy: {best_row['Accuracy']:.3f}")
    print(f"Balanced Accuracy: {best_row['BalancedAcc']:.3f}")
    print("\nMejores hiperparámetros:")
    for k, v in best_params.items():
        print(f"  - {k}: {v}")

    return best_model


# ============================================================
# NUEVAS TAREAS DE IMPORTANCIA DE VARIABLES
# ============================================================

@task
def analyze_feature_importance_rf_classification(df, feature_cols, quartile_col, preprocessor):
    print("\n[Feature Importance] Random Forest (Clasificación) ...")
    X = df[feature_cols]
    y = df[quartile_col]

    if y.min() > 0:
        y = y - y.min()
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)

    pipe = ImbPipeline([
        ("preproc", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=N_JOBS))
    ])
    pipe.fit(X, y)

    # >>> NEW (robust)
    ct = pipe.named_steps["preproc"]
    num_features = ct.transformers_[0][2] if ct.transformers_[0][2] is not None else []
    cat_features = []
    try:
        ohe = ct.named_transformers_["cat"].named_steps["onehot"]
        cat_input_cols = ct.transformers_[1][2] if len(ct.transformers_) > 1 else []
        cat_features = list(ohe.get_feature_names_out(cat_input_cols))
    except Exception as e:
        print(f"[WARN] No se pudieron extraer nombres categóricos: {e}")
    feature_names = np.array(list(num_features) + list(cat_features))
    importances = pipe.named_steps["model"].feature_importances_
    n_min = min(len(feature_names), len(importances))
    feature_names = feature_names[:n_min]
    importances = importances[:n_min]
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)
    # <<< NEW

    csv_path = os.path.join(OUTPUT_DIR, "rf_classifier_feature_importance.csv")
    png_path = os.path.join(OUTPUT_DIR, "rf_classifier_feature_importance.png")
    imp_df.to_csv(csv_path, index=False)
    plt.figure(figsize=(10, 7))
    plt.barh(imp_df.head(20)["Feature"][::-1], imp_df.head(20)["Importance"][::-1], color="steelblue")
    plt.title("Importancia de variables - Random Forest (Clasificación)")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Guardado: {csv_path} | {png_path}")


@task
def analyze_feature_importance_rf_regression(df, feature_cols, preprocessor):
    print("\n[Feature Importance] Random Forest (Regresión) ...")
    X = df[feature_cols]
    y = df["total_score"]

    pipe = Pipeline([
        ("preproc", preprocessor),
        ("model", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=N_JOBS))
    ])
    pipe.fit(X, y)

    # >>> NEW (robust)
    ct = pipe.named_steps["preproc"]
    num_features = ct.transformers_[0][2] if ct.transformers_[0][2] is not None else []
    cat_features = []
    try:
        ohe = ct.named_transformers_["cat"].named_steps["onehot"]
        cat_input_cols = ct.transformers_[1][2] if len(ct.transformers_) > 1 else []
        cat_features = list(ohe.get_feature_names_out(cat_input_cols))
    except Exception as e:
        print(f"[WARN] No se pudieron extraer nombres categóricos: {e}")
    feature_names = np.array(list(num_features) + list(cat_features))
    importances = pipe.named_steps["model"].feature_importances_
    n_min = min(len(feature_names), len(importances))
    feature_names = feature_names[:n_min]
    importances = importances[:n_min]
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)
    # <<< NEW

    csv_path = os.path.join(OUTPUT_DIR, "rf_regressor_feature_importance.csv")
    png_path = os.path.join(OUTPUT_DIR, "rf_regressor_feature_importance.png")
    imp_df.to_csv(csv_path, index=False)
    plt.figure(figsize=(10, 7))
    plt.barh(imp_df.head(20)["Feature"][::-1], imp_df.head(20)["Importance"][::-1], color="darkorange")
    plt.title("Importancia de variables - Random Forest (Regresión)")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Guardado: {csv_path} | {png_path}")


# ============================================================
# FLOW PRINCIPAL
# ============================================================

@flow(name="academic_pipeline_prefect")
def academic_pipeline():
    df = load_data(INPUT_PATH)
    df, feature_cols, quartile_col, preprocessor = preprocess_data(df)
    df_clf_results = train_classification(df, feature_cols, quartile_col, preprocessor)
    df_reg_results = train_regression(df, feature_cols, preprocessor)
    best_model_report(df_clf_results)
    analyze_feature_importance_rf_classification(df, feature_cols, quartile_col, preprocessor)
    analyze_feature_importance_rf_regression(df, feature_cols, preprocessor)
    print("\nPipeline completo finalizado correctamente.")


if __name__ == "__main__":
    academic_pipeline()