# Stock-Market-Analysis-and-Prediction-ML-Model

# importing Libraries
import pandas as pd

import numpy as np

import yfinance as yf

import matplotlib.pyplot as plt

import shap

import joblib

from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler

from indicators import add_indicators  # your existing indicators module

# --- Stock Categories and Symbols ---
# In a real-world scenario, you'd expand this with more categories (e.g., 'Banking', 'IT', 'Pharma')
# and populate them with relevant symbols.
STOCK_CATEGORIES = {
   
'Nifty 50': 

{
        'Adani Enterprises': 'ADANIENT.NS',
        'Adani Ports & SEZ': 'ADANIPORTS.NS',
        'Apollo Hospitals': 'APOLLOHOSP.NS',
        'Asian Paints': 'ASIANPAINT.NS',
        'Axis Bank': 'AXISBANK.NS',
        'Bajaj Auto': 'BAJAJ-AUTO.NS',
        'Bajaj Finance': 'BAJFINANCE.NS',
        'Bajaj Finserv': 'BAJAJFINSV.NS',
        'Bharat Electronics': 'BEL.NS',
        'Bharti Airtel': 'BHARTIARTL.NS',
        'Cipla': 'CIPLA.NS',
        'Coal India': 'COALINDIA.NS',
        "Dr. Reddy's Laboratories": 'DRREDDY.NS',
        'Eicher Motors': 'EICHERMOT.NS',
        'Grasim Industries': 'GRASIM.NS',
        'HCLTech': 'HCLTECH.NS',
        'HDFC Bank': 'HDFCBANK.NS',
        'HDFC Life': 'HDFCLIFE.NS',
        'Hero MotoCorp': 'HEROMOTOCO.NS',
        'Hindalco Industries': 'HINDALCO.NS',
        'Hindustan Unilever': 'HINDUNILVR.NS',
        'ICICI Bank': 'ICICIBANK.NS',
        'IndusInd Bank': 'INDUSINDBK.NS',
        'Infosys': 'INFY.NS',
        'ITC': 'ITC.NS',
        'Jio Financial Services': 'JIOFIN.NS',
        'JSW Steel': 'JSWSTEEL.NS',
        'Kotak Mahindra Bank': 'KOTAKBANK.NS',
        'Larsen & Toubro': 'LT.NS',
        'Mahindra & Mahindra': 'M&M.NS',
        'Maruti Suzuki': 'MARUTI.NS',
        'Nestlé India': 'NESTLEIND.NS',
        'NTPC': 'NTPC.NS',
        'Oil and Natural Gas Corporation': 'ONGC.NS',
        'Power Grid': 'POWERGRID.NS',
        'Reliance Industries': 'RELIANCE.NS',
        'SBI Life Insurance Company': 'SBILIFE.NS',
        'Shriram Finance': 'SHRIRAMFIN.NS',
        'State Bank of India': 'SBIN.NS',
        'Sun Pharma': 'SUNPHARMA.NS',
        'Tata Consultancy Services': 'TCS.NS',
        'Tata Consumer Products': 'TATACONSUM.NS',
        'Tata Motors': 'TATAMOTORS.NS',
        'Tata Steel': 'TATASTEEL.NS',
        'Tech Mahindra': 'TECHM.NS',
        'Titan Company': 'TITAN.NS',
        'Trent': 'TRENT.NS',
        'UltraTech Cement': 'ULTRACEMCO.NS',
        'Wipro': 'WIPRO.NS'
    }
}

def prepare_data(stock_symbol, start_date="2010-01-01", end_date=datetime.now().strftime('%Y-%m-%d')):

 """Downloads data, adds indicators, creates lagged features, and prepares target variable."""
 
print(f"📥 Downloading data for {stock_symbol}...")

 df = yf.download(stock_symbol, start=start_date, end=end_date, auto_adjust=True)
 
if df.empty:

 print(f"Skipping {stock_symbol} due to no data found.")
        return None, None, None, None

    # Add technical indicators using your module
    
  df = add_indicators(df)

    # Create additional features - lagged features for momentum capture
    
  df['Close_lag_1'] = df['Close'].shift(1)
  
  df['Close_lag_2'] = df['Close'].shift(2)
  
  df['Close_lag_3'] = df['Close'].shift(3)
    
    # Target: classify next day's return with threshold
    # 1 if next day's close is > 1% higher, 0 if < -1% lower, NaN otherwise
    
  df['Target'] = np.where(
  
 (df['Close'].shift(-1) - df['Close']) / df['Close'] > 0.01, 1,
 
 np.where((df['Close'].shift(-1) - df['Close']) / df['Close'] < -0.01, 0, np.nan)
 
)

   df.dropna(inplace=True)
   
 features =
 [
        'RSI', 'SMA_20', 'SMA_50', 'EMA_20', 'MACD', 'MACD_Signal',
        'Stoch_K', 'Stoch_D', 'Volume_SMA_20', 'Daily_Return', 'Momentum',
        'BB_Upper', 'BB_Lower', 'ATR',
        'Close_lag_1', 'Close_lag_2', 'Close_lag_3'
    ]

    # Filter features that might not have been created by add_indicators (e.g., if data is too short)
    available_features = [f for f in features if f in df.columns]
    
   X = df[available_features]
   
 y = df['Target']
 
 if X.empty or y.empty:
 
 print(f"Skipping {stock_symbol} due to insufficient data after feature creation or NaN handling.")
        return None, None, None, None

    # Remove highly correlated features (>0.9) to reduce multicollinearity
    
  corr_matrix = X.corr().abs()
  
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

 to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
 
if to_drop:

  print(f"🧹 Removing correlated features for {stock_symbol}: {to_drop}")
        X = X.drop(columns=to_drop)

    # Scale features
    
   scaler = StandardScaler()
   
  X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
  
  return X_scaled, y, df, scaler

def train_model(X, y, model_save_path='stock_model_improved.joblib',
                feature_importance_path='feature_importance_improved.png',
                shap_summary_path='shap_summary_improved.png'):
                
"""Trains an XGBoost model using TimeSeriesSplit and GridSearchCV."""

print("🧪 Training model with TimeSeriesSplit and GridSearchCV...")

  tscv = TimeSeriesSplit(n_splits=5)
  
 params =
 {
        'n_estimators': [300, 500],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.7, 1.0],
        'gamma': [0, 1],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 2],
    }
    
xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

 grid = GridSearchCV(xgb, params, cv=tscv, scoring='accuracy', verbose=0, n_jobs=-1)
 
grid.fit(X, y)

 print(f"✅ Best Params: {grid.best_params_}")
 
best_model = grid.best_estimator_

    # Last fold train-test split for performance metrics
    
  train_idx, test_idx = list(tscv.split(X))[-1]
  
   X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
   
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

   y_pred = best_model.predict(X_test)
   
acc = accuracy_score(y_test, y_pred)

print(f"🎯 Model Accuracy on test fold: {acc * 100:.2f}%")

print("\n📊 Classification Report:")
   
  print(classification_report(y_test, y_pred))

    # Save model
    
  joblib.dump(best_model, model_save_path)
  
print(f"💾 Model saved as {model_save_path}")

   # Feature importance plot
   
  plt.figure(figsize=(10, 6))
  
feature_imp = pd.Series(best_model.feature_importances_, index=X.columns)
    feature_imp.sort_values().plot(kind='barh')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(feature_importance_path)
    plt.close()
    print("📈 Feature importance plot saved.")

    # SHAP explainability
    
  print("🔍 Generating SHAP summary plot...")
  
    # SHAP can sometimes have issues with older Python/Numpy versions.
    # If shap.TreeExplainer gives an error, consider updating your environment or checking compatibility.
    try:
    
  explainer = shap.TreeExplainer(best_model)
  
shap_values = explainer.shap_values(X_test)
        plt.figure()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(shap_summary_path)
        plt.close()
        print("📊 SHAP summary plot saved.")
    except Exception as e:
        print(f"⚠️ SHAP plot generation failed: {e}. Skipping SHAP plot.")
    return best_model, X.columns

def predict_next_day(model, scaler, latest_data, feature_names):
    """Predicts the next day's stock movement and provides investment advice."""
    
    # Ensure latest_data is a DataFrame with correct index and columns
    latest_features = latest_data[feature_names].to_frame().T
    
    # Scale only the features, not the target or other columns
    
  latest_scaled = pd.DataFrame(scaler.transform(latest_features), 
                                 columns=feature_names, 
                                 index=latest_features.index)
    prob = model.predict_proba(latest_scaled)[0]
    pred = model.predict(latest_scaled)[0]
    bull_prob = prob[1] * 100
    bear_prob = prob[0] * 100
    if pred == 1:
        print(f"\n📈 Prediction: **Bullish market tomorrow** with {bull_prob:.2f}% confidence.")
    else:
        print(f"\n📉 Prediction: **Bearish market tomorrow** with {bear_prob:.2f}% confidence.")
    if pred == 1 and bull_prob > 70:
        advice = "STRONG BUY"
    elif pred == 1 and bull_prob > 60:
        advice = "BUY"
    elif pred == 0 and bear_prob > 70:
        advice = "STRONG SELL"
    elif pred == 0 and bear_prob > 60:
        advice = "SELL"
    else:
        advice = "HOLD (Market Neutral)"
    print(f"💡 Investment Advice: **{advice}**")
    print(f"🔍 Probabilities - Bullish: {bull_prob:.2f}%, Bearish: {bear_prob:.2f}%")
    return pred, bull_prob

def backtest_model(model, scaler, df, feature_names, days=30, backtest_results_path="backtest_results_improved.csv"):
    """Performs a backtest on the model's performance over the last 'days'."""
    print(f"\n🔄 Running backtest on last {days} days...")
    if len(df) < days:
        print(f"Insufficient data for backtest. Need at least {days} days, but only {len(df)} available.")
        return
    test_df = df.iloc[-days:].copy() # Use .copy() to avoid SettingWithCopyWarning
    X_bt = test_df[feature_names]
    X_bt_scaled = scaler.transform(X_bt)
    y_bt = test_df['Target']
    y_pred = model.predict(X_bt_scaled)
    acc = accuracy_score(y_bt, y_pred)
    cm = confusion_matrix(y_bt, y_pred)
    results = pd.DataFrame({
        'Date': test_df.index,
        'Actual': y_bt.values,
        'Predicted': y_pred,
    })
    results['Match'] = results['Actual'] == results['Predicted']
    print(f"\n📅 Backtest Accuracy (Last {days} Days): {acc * 100:.2f}%")
    print("📊 Confusion Matrix:")
    print(cm)
    print("\n📝 Sample Prediction Table (Last 10 Days):")
    print(results.tail(10).to_string()) # Using to_string() for better console display
    results.to_csv(backtest_results_path, index=False)
    print(f"📁 Backtest results saved to {backtest_results_path}")

def recommend_top_stocks(category_name, num_recommendations=5):
    """
    Trains models for all stocks in a category and recommends the top N bullish stocks.
    """
    print(f"\n✨ Identifying top {num_recommendations} stocks for investment in **{category_name}**...")
    category_symbols = STOCK_CATEGORIES[category_name]
    stock_predictions = []
    for name, symbol in category_symbols.items():
        print(f"\nProcessing {name} ({symbol})...")
        X, y, df, scaler = prepare_data(symbol)
        if X is None or y is None:
            print(f"Could not prepare data for {name}. Skipping.")
            continue
        try:
            model, features = train_model(X, y, 
                                          model_save_path=f'models/{symbol.replace(".NS", "")}_model.joblib',
                                          feature_importance_path=f'plots/{symbol.replace(".NS", "")}_feature_importance.png',
                                          shap_summary_path=f'plots/{symbol.replace(".NS", "")}_shap_summary.png')
            latest = df.iloc[-1]
            prob = model.predict_proba(scaler.transform(latest[features].to_frame().T))[0]
            bull_prob = prob[1] * 100
                stock_predictions.append({
                'Company': name,
                'Symbol': symbol,
                'Bullish_Confidence': bull_prob,
                'Prediction': 'Bullish' if model.predict(scaler.transform(latest[features].to_frame().T))[0] == 1 else 'Bearish'
            })
        except Exception as e:
            print(f"❗ Error processing {name} ({symbol}): {e}. Skipping this stock for recommendations.")
            continue

    # Sort by bullish confidence in descending order
    
  stock_predictions.sort(key=lambda x: x['Bullish_Confidence'], reverse=True)
    print(f"\n--- Top {num_recommendations} Stocks for {category_name} ---")
    recommended_count = 0
    for i, stock in enumerate(stock_predictions):
        if stock['Prediction'] == 'Bullish':
            print(f"{recommended_count + 1}. **{stock['Company']}** ({stock['Symbol']}) - Prediction: {stock['Prediction']}, Confidence: {stock['Bullish_Confidence']:.2f}%")
            recommended_count += 1
        if recommended_count >= num_recommendations:
            break
    if recommended_count == 0:
        print(f"No bullish recommendations found for {category_name} today. Consider holding or waiting for better opportunities.")


if __name__ == "__main__":
    print("--- Stock Market Prediction and Analysis Tool ---")

    # --- Step 1: Choose a Category ---
    
  print("\nSelect a Stock Category:")
    categories_list = list(STOCK_CATEGORIES.keys())
    for i, category in enumerate(categories_list, 1):
        print(f"{i}. {category}")

  selected_category_name = None
    while selected_category_name is None:
        try:
            category_choice = int(input("\nEnter the number of the category: "))
            if 1 <= category_choice <= len(categories_list):
                selected_category_name = categories_list[category_choice - 1]
                print(f"\nSelected Category: **{selected_category_name}**")
            else:
                print("❌ Invalid category number. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number.")

    # --- Step 2: Ask for overall category recommendation or specific stock analysis ---
    
  print("\nWhat would you like to do?")
    print("1. Get **Top 5 Investment Recommendations** for this category today.")
    print("2. Analyze a **Specific Stock** within this category.")
    user_action_choice = None
    while user_action_choice is None:
        try:
            action_choice = int(input("Enter your choice (1 or 2): "))
            if action_choice in [1, 2]:
                user_action_choice = action_choice
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("❌ Please enter a valid number.")
    if user_action_choice == 1:
        recommend_top_stocks(selected_category_name)
    else: # user_action_choice == 2
    
        # --- Step 3: Choose a Specific Stock within the Selected Category ---
        
  print(f"\n📊 Stocks in {selected_category_name}:")
        company_list = list(STOCK_CATEGORIES[selected_category_name].items())
        for i, (name, symbol) in enumerate(company_list, 1):
            print(f"{i}. {name} ({symbol})")
        selected_company_name = None
        stock_symbol = None
        while stock_symbol is None:
            try:
                stock_choice = int(input(f"\nSelect a company from {selected_category_name} by number (e.g., 1): "))
                if 1 <= stock_choice <= len(company_list):
                    selected_company_name, stock_symbol = company_list[stock_choice - 1]
                    print(f"\n📦 You selected: **{selected_company_name}** ({stock_symbol})")
                else:
                    print("❌ Invalid selection number. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number.")

        # --- Step 4: Process the selected stock ---
        
  X, y, df, scaler = prepare_data(stock_symbol)
        if X is None or y is None:
            print(f"❗ Could not proceed with {selected_company_name} due to data preparation issues. Exiting.")
        else:
            model, features = train_model(X, y)
            latest = df.iloc[-1]
            predict_next_day(model, scaler, latest, features)
            backtest_model(model, scaler, df, features, days=30) 
    print("\n--- Analysis Complete ---")
