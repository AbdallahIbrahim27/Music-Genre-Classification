"""
tabular_approach.py
-------------------
Approach 1: MFCC features → Scikit-learn classifiers
  • Random Forest
  • Support Vector Machine (RBF kernel)

Outputs:
  - Accuracy, classification report, confusion matrix plot
  - Best model saved as tabular_best_model.pkl
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix)
from sklearn.pipeline        import Pipeline

from feature_extraction import build_feature_dataframe, GENRES

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT   = 'data'
CSV_CACHE   = 'features.csv'
RANDOM_SEED = 42
TEST_SIZE   = 0.20
# ──────────────────────────────────────────────────────────────────────────────


def load_or_extract(data_root: str, csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        print(f"Loading cached features from {csv_path} ...")
        return pd.read_csv(csv_path)
    print("Extracting features (this may take a few minutes) ...")
    return build_feature_dataframe(data_root, csv_path)


def prepare_data(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in ('genre', 'filename')]
    X = df[feature_cols].values
    le = LabelEncoder()
    y  = le.fit_transform(df['genre'].values)
    return X, y, le, feature_cols


def plot_confusion_matrix(cm, classes, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True',      fontsize=12)
    ax.set_title(title,        fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {out_path}")


def evaluate(model_name, pipeline, X_train, X_test, y_train, y_test, le):
    print(f"\n{'─'*50}")
    print(f"  {model_name}")
    print(f"{'─'*50}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy : {acc:.4f} ({acc*100:.2f}%)")

    # Cross-validation on training set
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"  CV accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("\n  Classification report:")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_,
                                digits=3))

    cm = confusion_matrix(y_test, y_pred)
    out = f"cm_{model_name.lower().replace(' ', '_')}.png"
    plot_confusion_matrix(cm, le.classes_, f"{model_name} — Confusion Matrix", out)

    return acc, pipeline


def main():
    os.makedirs('outputs', exist_ok=True)

    # 1. Load / extract features
    df = load_or_extract(DATA_ROOT, CSV_CACHE)
    print(f"\nDataset shape: {df.shape}")
    print(f"Genre distribution:\n{df['genre'].value_counts()}\n")

    # 2. Prepare
    X, y, le, feature_cols = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

    print(f"Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")

    # 3. Models
    results = {}

    # Random Forest (no scaling needed)
    rf_pipe = Pipeline([
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ))
    ])
    acc_rf, rf_pipe = evaluate('Random Forest', rf_pipe,
                               X_train, X_test, y_train, y_test, le)
    results['Random Forest'] = acc_rf

    # SVM (needs scaling)
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=10, gamma='scale',
                    decision_function_shape='ovr',
                    random_state=RANDOM_SEED))
    ])
    acc_svm, svm_pipe = evaluate('SVM (RBF)', svm_pipe,
                                  X_train, X_test, y_train, y_test, le)
    results['SVM (RBF)'] = acc_svm

    # 4. Save best model
    best_name, best_acc = max(results.items(), key=lambda x: x[1])
    best_model = rf_pipe if best_name == 'Random Forest' else svm_pipe
    model_path = 'tabular_best_model.pkl'
    joblib.dump({'model': best_model, 'label_encoder': le,
                 'feature_cols': feature_cols}, model_path)
    print(f"\n🏆 Best model: {best_name}  ({best_acc*100:.2f}%)")
    print(f"   Saved → {model_path}")

    # 5. Feature importance (RF only)
    rf_clf = rf_pipe.named_steps['clf']
    importances = pd.Series(rf_clf.feature_importances_, index=feature_cols)
    top20 = importances.nlargest(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    top20.sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Top-20 Feature Importances (Random Forest)', fontsize=14)
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.close()
    print("  Feature importance plot saved → feature_importance.png")

    # 6. Summary bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(results.keys(), [v*100 for v in results.values()],
                  color=['#2196F3', '#FF5722'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Tabular Approach — Model Comparison')
    for bar, val in zip(bars, results.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val*100:.1f}%', ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('tabular_comparison.png', dpi=150)
    plt.close()
    print("  Comparison chart saved → tabular_comparison.png")

    return results


if __name__ == '__main__':
    main()
