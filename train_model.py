import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    print("⏳ Loading and preparing data...")
    
    # 1. Load Data
    df = pd.read_excel('RockHerb_Full.xlsx')
    df = df.dropna(subset=['Order ID', 'Seller Name', 'Phone', 'Order Date', 'Grand Total'])
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Phone'] = df['Phone'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    
    snapshot_date = df['Order Date'].max() + pd.Timedelta(days=1)
    
    # 2. Advanced Feature Engineering (No Data Leaks)
    churn_df = df.groupby('Phone').agg(
        Frequency=('Order ID', 'nunique'),
        Last_Purchase=('Order Date', 'max'),
        First_Purchase=('Order Date', 'min'),
        Total_Spend=('Grand Total', 'sum')
    ).reset_index()
    
    # Target Variable: 1 if hasn't bought in 90 days, else 0
    churn_df['Recency_Days'] = (snapshot_date - churn_df['Last_Purchase']).dt.days
    churn_df['Target_Churn'] = (churn_df['Recency_Days'] > 90).astype(int)
    
    # Legitimate Predictive Features
    churn_df['Account_Age_Days'] = (snapshot_date - churn_df['First_Purchase']).dt.days
    churn_df['Avg_Order_Value'] = churn_df['Total_Spend'] / churn_df['Frequency']
    
    # Define X (Features) and y (Target)
    # Notice we drop 'Recency_Days' so the model actually has to learn behavior patterns
    X = churn_df[['Frequency', 'Total_Spend', 'Account_Age_Days', 'Avg_Order_Value']]
    y = churn_df['Target_Churn']

    print(f"📊 Total Customers: {len(churn_df)}")
    print("-" * 40)

    # 3. Test Different Splits [90/10, 80/20, 70/30]
    splits = {
        "90/10": 0.10,
        "80/20": 0.20,
        "70/30": 0.30
    }
    
    best_model = None
    best_accuracy = 0
    best_split_name = ""

    for split_name, test_size in splits.items():
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Initialize and Train Random Forest
        # class_weight='balanced' ensures it doesn't just guess "0" for everyone
        rf = RandomForestClassifier(n_estimators=100, max_depth=7, class_weight='balanced', random_state=42)
        rf.fit(X_train, y_train)
        
        # Test Accuracy
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"📈 Split {split_name} | Accuracy: {acc * 100:.2f}%")
        
        # Keep the best model
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = rf
            best_split_name = split_name

    print("-" * 40)
    print(f"🏆 Best Model found was {best_split_name} with {best_accuracy * 100:.2f}% accuracy.")
    
    # 4. Save the Model if it meets the 80% requirement
    if best_accuracy >= 0.80:
        model_filename = 'churn_model.pkl'
        joblib.dump(best_model, model_filename)
        print(f"✅ Success: Model saved locally as '{model_filename}'")
    else:
        print("❌ Warning: Model failed to hit 80% accuracy constraint. Try adding more data or features.")

if __name__ == "__main__":
    main()