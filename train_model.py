import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import joblib
import warnings
warnings.filterwarnings('ignore')

CHURN_WINDOW_DAYS = 90
MIN_ACCOUNT_AGE_DAYS = 90  # customers younger than this cannot possibly be "churned" yet


def main():
    print("Loading and preparing data...")

    # 1. Load Data
    df = pd.read_excel('RockHerb_Full.xlsx')
    df = df.dropna(subset=['Order ID', 'Seller Name', 'Phone', 'Order Date', 'Grand Total'])
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Phone'] = df['Phone'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

    snapshot_date = df['Order Date'].max() + pd.Timedelta(days=1)

    # 2. Aggregate to customer level
    churn_df = df.groupby('Phone').agg(
        Frequency=('Order ID', 'nunique'),
        Last_Purchase=('Order Date', 'max'),
        First_Purchase=('Order Date', 'min'),
        Total_Spend=('Grand Total', 'sum')
    ).reset_index()

    churn_df['Recency_Days'] = (snapshot_date - churn_df['Last_Purchase']).dt.days
    churn_df['Account_Age_Days'] = (snapshot_date - churn_df['First_Purchase']).dt.days
    churn_df['Avg_Order_Value'] = churn_df['Total_Spend'] / churn_df['Frequency']
    churn_df['Target_Churn'] = (churn_df['Recency_Days'] > CHURN_WINDOW_DAYS).astype(int)

    print(f"Total customers (raw): {len(churn_df)}")
    print(f"Raw churn rate: {churn_df['Target_Churn'].mean() * 100:.2f}% "
          f"(this is your accuracy baseline - beat THIS, not 80%)")

    # ------------------------------------------------------------------
    # FIX #1: Remove customers who cannot possibly be labeled churned.
    # Anyone younger than the churn window is guaranteed Target_Churn=0
    # by arithmetic (Recency_Days <= Account_Age_Days), not behavior.
    # Leaving them in lets the model "solve" churn just by reading age.
    # ------------------------------------------------------------------
    before = len(churn_df)
    churn_df = churn_df[churn_df['Account_Age_Days'] >= MIN_ACCOUNT_AGE_DAYS].copy()
    print(f"Removed {before - len(churn_df)} customers too new to have a valid label "
          f"({MIN_ACCOUNT_AGE_DAYS}-day floor)")
    print(f"Modeling population: {len(churn_df)} | "
          f"churn rate now: {churn_df['Target_Churn'].mean() * 100:.2f}%")
    print("-" * 60)

    X = churn_df[['Frequency', 'Total_Spend', 'Account_Age_Days', 'Avg_Order_Value']]
    y = churn_df['Target_Churn']

    # ------------------------------------------------------------------
    # FIX #2: One honest train/test split, held out until the very end.
    # Model selection happens via cross-validation on the training data
    # only - never by peeking at test-set scores across several splits.
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=7, class_weight='balanced', random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(rf, X_train, y_train, cv=cv, scoring='roc_auc')
    cv_ap = cross_val_score(rf, X_train, y_train, cv=cv, scoring='average_precision')

    print(f"5-fold CV ROC-AUC:  {cv_auc.mean():.3f} (+/- {cv_auc.std():.3f})")
    print(f"5-fold CV PR-AUC:   {cv_ap.mean():.3f} (+/- {cv_ap.std():.3f})")
    print("-" * 60)

    # Fit once on full training set, evaluate once on the untouched test set
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    print("Held-out test set performance:")
    print(classification_report(y_test, y_pred, target_names=['Active(0)', 'Churned(1)']))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    print(f"Test PR-AUC:  {average_precision_score(y_test, y_proba):.3f}")
    print("-" * 60)

    feature_importance = dict(zip(X.columns, rf.feature_importances_))
    print("Feature importances:", {k: round(v, 3) for k, v in feature_importance.items()})
    print("-" * 60)

    # ------------------------------------------------------------------
    # FIX #3: Judge on recall for the Active class (the commercially
    # useful signal - who's actually worth marketing to) and ROC-AUC,
    # not raw accuracy, which is meaningless under this much imbalance.
    # ------------------------------------------------------------------
    active_recall = classification_report(
        y_test, y_pred, target_names=['Active(0)', 'Churned(1)'], output_dict=True
    )['Active(0)']['recall']

    MIN_ACTIVE_RECALL = 0.70
    MIN_AUC = 0.75

    test_auc = roc_auc_score(y_test, y_proba)
    if active_recall >= MIN_ACTIVE_RECALL and test_auc >= MIN_AUC:
        model_filename = 'churn_model.pkl'
        joblib.dump(rf, model_filename)
        print(f"Saved '{model_filename}' | Active recall: {active_recall:.2f}, ROC-AUC: {test_auc:.2f}")
    else:
        print(f"Model did not meet bar (Active recall {active_recall:.2f} / ROC-AUC {test_auc:.2f}). "
              f"Needs more features or a different churn definition before shipping.")


if __name__ == "__main__":
    main()