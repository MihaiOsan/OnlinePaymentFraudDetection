import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Încarcă setul de date
file_path = 'PS_20174392719_1491204439457_log.csv'
df = pd.read_csv(file_path)

# Eliminăm coloanele `nameOrig`, `nameDest` și `newbalanceDest`
df.drop(columns=['nameOrig', 'nameDest', 'newbalanceDest'], inplace=True)

# Definirea coloanelor pentru codificare și normalizare
categorical_features = ['type']
numeric_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest']

# Crearea unui pipeline pentru preprocesare
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Aplicăm preprocesarea pe întreg setul de date înainte de a împărți datele
X = df.drop('isFraud', axis=1)
y = df['isFraud']
print("Aplicăm preprocesarea inițială pe date...")
X = preprocessor.fit_transform(X)
print("Preprocesarea inițială completată.")

# Împărțirea setului de date în antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2001, stratify=y)
print("Împărțirea setului de date completată.")

# Definirea modelelor
models = {
    'RandomForest': RandomForestClassifier(random_state=2001),
    'LogisticRegression': LogisticRegression(random_state=2001, max_iter=1000),
    'GradientBoosting': GradientBoostingClassifier(random_state=2001)
}

# Funcție pentru antrenarea și evaluarea modelului
def train_and_evaluate(X_train, y_train, X_test, y_test, model, balance_method=None):
    if balance_method:
        pipeline = ImbPipeline([
            ('balance', balance_method),
            ('classifier', model)
        ])
    else:
        pipeline = Pipeline([
            ('classifier', model)
        ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Model: {model.__class__.__name__}, Balance Method: {balance_method.__class__.__name__ if balance_method else 'None'}")
    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc}")
    print("--------------------------------------------------\n")

    return report, accuracy, auc

# Evaluarea fiecărui model folosind tehnici de echilibrare
results = []
for balance_method, method_name in zip([None, SMOTE(random_state=2001), RandomOverSampler(random_state=2001), RandomUnderSampler(random_state=2001)],
                                       ['No balancing', 'SMOTE', 'Oversampling', 'Undersampling']):
    for model_name, model in models.items():
        print(f"\nEvaluare pentru modelul {model_name} cu metoda de echilibrare {method_name}")
        report, accuracy, auc = train_and_evaluate(X_train, y_train, X_test, y_test, model, balance_method)
        report_df = pd.DataFrame(report).transpose()
        report_df['model'] = model_name
        report_df['balance_method'] = method_name
        report_df['accuracy'] = accuracy
        report_df['auc'] = auc
        results.append(report_df)

# Concatenăm rezultatele într-un singur DataFrame
final_results = pd.concat(results, ignore_index=True)

# Salvează rezultatele într-un fișier CSV pentru utilizarea ulterioară
final_results.to_csv('final_results.csv', index=False)

# Afișăm rezultatele
print("Evaluarea completată. Rezultatele finale:")
print(final_results)

# Funcție pentru generarea ploturilor
def plot_results(final_results, metric):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=final_results, x='model', y=metric, hue='balance_method')
    plt.title(f'Model Comparison by {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.legend(title='Balance Method')
    plt.show()

# Plot pentru acuratețe
plot_results(final_results, 'accuracy')

# Plot pentru AUC
plot_results(final_results, 'auc')
