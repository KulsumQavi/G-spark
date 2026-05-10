"""
FarmLinks ML Training Script
Trains 3 models: Driver Matcher, Price Forecaster, Spoilage Classifier
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
import os

os.makedirs('ml/models', exist_ok=True)

print("🤖 FarmLinks ML Training Started...\n")

# ==========================================
# MODEL 1: DRIVER MATCH SCORE PREDICTOR
# ==========================================
print("=" * 50)
print("📊 MODEL 1: Driver Match Score Predictor")
print("=" * 50)

np.random.seed(42)
n_samples = 5000

# Features: rating, points, capacity, distance, urgency, past_completed
rating = np.random.uniform(3.0, 5.0, n_samples)
points = np.random.randint(0, 5000, n_samples)
capacity = np.random.choice([500, 800, 1000, 1500, 2000, 3000], n_samples)
distance = np.random.uniform(1, 100, n_samples)
urgency = np.random.choice([1, 2, 3], n_samples)
past_completed = np.random.randint(0, 200, n_samples)

# Target: match score (synthetic formula with noise)
match_score = (
    rating * 18 +
    np.log1p(points) * 8 +
    (capacity / 100) * 1.2 -
    distance * 0.4 +
    urgency * 5 +
    past_completed * 0.2 +
    np.random.normal(0, 5, n_samples)
)
match_score = np.clip(match_score, 0, 100)

X1 = pd.DataFrame({
    'rating': rating,
    'points': points,
    'capacity': capacity,
    'distance': distance,
    'urgency': urgency,
    'past_completed': past_completed
})
y1 = match_score

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

model1 = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
model1.fit(X_train, y_train)

predictions = model1.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"✅ Trained on {n_samples} samples")
print(f"📈 Accuracy (MAE): {mae:.2f}")
print(f"💾 Saving to ml/models/driver_matcher.pkl")
joblib.dump(model1, 'ml/models/driver_matcher.pkl')


# ==========================================
# MODEL 2: PRICE FORECASTER
# ==========================================
print("\n" + "=" * 50)
print("📊 MODEL 2: Crop Price Forecaster")
print("=" * 50)

n_samples = 8000

# Features: crop_id, day_of_year, demand_level, supply_level, weather_score, season
crops = ['Tomatoes', 'Onions', 'Potatoes', 'Chillies', 'Wheat', 'Rice', 'Mangoes', 'Bananas']
crop_base_prices = {'Tomatoes': 35, 'Onions': 28, 'Potatoes': 22, 'Chillies': 80,
                    'Wheat': 32, 'Rice': 45, 'Mangoes': 120, 'Bananas': 40}

data = []
for _ in range(n_samples):
    crop = np.random.choice(crops)
    crop_id = crops.index(crop)
    base_price = crop_base_prices[crop]
    
    day_of_year = np.random.randint(1, 366)
    demand = np.random.uniform(0.3, 1.5)
    supply = np.random.uniform(0.4, 1.6)
    weather = np.random.uniform(0.5, 1.2)
    season = (day_of_year % 365) / 365
    
    # Synthetic price formula
    price = base_price * (
        1 + 
        0.3 * demand - 
        0.25 * supply + 
        0.15 * np.sin(season * 2 * np.pi) + 
        0.1 * weather +
        np.random.normal(0, 0.05)
    )
    price = max(price, base_price * 0.5)
    
    data.append([crop_id, day_of_year, demand, supply, weather, season, price])

df = pd.DataFrame(data, columns=['crop_id', 'day', 'demand', 'supply', 'weather', 'season', 'price'])

X2 = df.drop('price', axis=1)
y2 = df['price']

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

model2 = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
model2.fit(X_train, y_train)

predictions = model2.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"✅ Trained on {n_samples} samples")
print(f"📈 Accuracy (MAE): ₹{mae:.2f}")
print(f"💾 Saving to ml/models/price_forecaster.pkl")
joblib.dump(model2, 'ml/models/price_forecaster.pkl')
joblib.dump(crops, 'ml/models/crop_list.pkl')


# ==========================================
# MODEL 3: SPOILAGE RISK CLASSIFIER
# ==========================================
print("\n" + "=" * 50)
print("📊 MODEL 3: Spoilage Risk Classifier")
print("=" * 50)

n_samples = 6000

# Features: temperature, humidity, hours_since_pickup, crop_perishability, distance_remaining
temp = np.random.uniform(0, 40, n_samples)
humidity = np.random.uniform(30, 100, n_samples)
hours = np.random.uniform(0, 48, n_samples)
perishability = np.random.choice([1, 2, 3, 4, 5], n_samples)  # 1=low, 5=high
distance = np.random.uniform(0, 200, n_samples)

# Synthetic spoilage logic
spoilage_score = (
    np.maximum(0, temp - 15) * 0.5 +
    np.maximum(0, humidity - 80) * 0.1 +
    hours * 0.4 +
    perishability * 2 +
    distance * 0.05 +
    np.random.normal(0, 1, n_samples)
)

# Classify: 0=safe, 1=warning, 2=critical
labels = np.where(spoilage_score < 8, 0, np.where(spoilage_score < 15, 1, 2))

X3 = pd.DataFrame({
    'temperature': temp,
    'humidity': humidity,
    'hours_since_pickup': hours,
    'perishability': perishability,
    'distance_remaining': distance
})
y3 = labels

X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.2, random_state=42)

model3 = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
model3.fit(X_train, y_train)

predictions = model3.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"✅ Trained on {n_samples} samples")
print(f"📈 Accuracy: {acc*100:.2f}%")
print(f"💾 Saving to ml/models/spoilage_classifier.pkl")
joblib.dump(model3, 'ml/models/spoilage_classifier.pkl')


# ==========================================
# DONE!
# ==========================================
print("\n" + "=" * 50)
print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
print("=" * 50)
print("\n📁 Models saved in ml/models/:")
print("   ├── driver_matcher.pkl")
print("   ├── price_forecaster.pkl")
print("   ├── crop_list.pkl")
print("   └── spoilage_classifier.pkl")
print("\n🚀 Ready to use in app.py!")