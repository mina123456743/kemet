from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import itertools

app = Flask(__name__)

# تحميل البيانات وتجهيزها
def load_and_prepare_data():
    # تحميل البيانات
    df = pd.read_excel("gr_pdata.xlsx")
    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)

    # تحويل المدن والمناطق لأرقام
    label_encoders = {}
    for col in ["City", "Region"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

# تدريب نموذج Random Forest
def train_model(df):
    X = df.drop(columns=["Price"])
    y = df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    return rf

# تحميل البيانات وتدريب النموذج
df, label_encoders = load_and_prepare_data()
rf = train_model(df)

# دالة توليد الاقتراحات
def suggest_regions(city, total_price_of_travel, num_regions, num_suggestions):
    try:
        city_encoded = label_encoders["City"].transform([city])[0]
    except ValueError:
        return []  # إذا كانت المدينة غير موجودة في البيانات

    city_mask = df["City"] == city_encoded
    df_filtered = df[city_mask].sort_values(by="Price", ascending=False)
    
    # إضافة تنبؤات الأسعار
    X_pred = df_filtered.drop(columns=["Price"])
    df_filtered["Predicted_Price"] = rf.predict(X_pred)
    
    # توليد جميع التركيبات الممكنة
    suggestions = []
    for r in range(1, num_regions + 1):
        for combo in itertools.combinations(df_filtered.itertuples(index=False), r):
            total_price = sum(region.Predicted_Price for region in combo)
            if total_price <= total_price_of_travel:
                regions = [region.Region for region in combo]
                suggestions.append((regions, total_price))
    
    # ترتيب الاقتراحات حسب السعر (من الأعلى للأدنى)
    suggestions.sort(key=lambda x: x[1], reverse=True)
    
    # تحضير النتائج للعرض
    result = []
    for regions, price in suggestions[:num_suggestions]:
        region_names = label_encoders["Region"].inverse_transform(regions)
        result.append({
            "regions": list(region_names),
            "price": float(price),
            "price_usd": round(float(price) / 51, 2)  # تحويل للدولار
        })
    
    return result

# نقطة النهاية للاستعلام
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # استقبال البيانات من Flutter
        data = request.json
        
        budget = int(data['budget'])
        needs_car = data['needs_car']
        needs_guide = data['needs_guide']
        city = data['city']
        num_trips = int(data['num_trips'])
        num_suggestions = int(data['num_suggestions'])
        
        # حساب تكلفة المرشد السياحي
        tour_guide_cost = 0
        if needs_guide.lower() == "yes":
            guide_days = int(data['guide_days'])
            tour_guide_cost = guide_days * 250
        
        # حساب تكلفة المواصلات
        transport_cost = 0
        if needs_car.lower() == "yes":
            city_prices = {
                "Cairo": 250, "Alex": 150, "Aswan": 200,
                "Luxor": 200, "Dahab": 350,
                "Sharm El-Sheikh": 400, "Hurghada": 250
            }
            transport_cost = num_trips * city_prices.get(city, 0)
        
        remaining_budget = budget - tour_guide_cost - transport_cost
        
        # توليد الاقتراحات
        suggestions = suggest_regions(city, remaining_budget, num_trips, num_suggestions)
        
        return jsonify({
            "status": "success",
            "suggestions": suggestions,
            "remaining_budget": remaining_budget,
            "transport_cost": transport_cost,
            "tour_guide_cost": tour_guide_cost
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

# نقطة نهاية للتحقق من عمل الخادم
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "active", "message": "Server is running"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)