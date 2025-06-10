import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, abort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

app = Flask(__name__)

# Global variables
df = None
vectorizer = None
tfidf_matrix = None

# Load the data manually
def load_data():
    global df, tfidf_matrix, vectorizer
    
    try:
        # Load data
        df = pd.read_excel("Egypt_hotels_data (1) (1).xlsx")
        
        # Clean and process the data (same as before)
        df.drop_duplicates(inplace=True)
        df.drop(columns=["Excluded Amenities", "Essential Info", "Link", "Reviews Breakdown"], inplace=True, errors='ignore')
        df['Check-In Time'] = df['Check-In Time'].str.replace('â€¯', ' ', regex=False)
        df['Check-Out Time'] = df['Check-Out Time'].str.replace('â€¯', ' ', regex=False)
        
        # Process price data
        for col in ["Rate per Night (Lowest)", "Rate per Night (Before Taxes and Fees)", "Total Rate (Lowest)", "Total Rate (Before Taxes and Fees)"]:
            if col in df.columns:
                df[col] = df[col].str.replace("EGPÂ", "", regex=False)
                df[col] = df[col].str.replace('\xa0', '', regex=False).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['Rate per Night'] = df['Rate per Night (Before Taxes and Fees)']
        df['Taxes and Fees'] = df['Total Rate (Lowest)'] - df['Total Rate (Before Taxes and Fees)']
        df['Total Rate'] = df['Total Rate (Lowest)']
        
        # Drop old columns
        df.drop(columns=['Rate per Night (Lowest)', 'Rate per Night (Before Taxes and Fees)', 'Total Rate (Lowest)', 'Total Rate (Before Taxes and Fees)'], inplace=True, errors='ignore')
        
        # Fill missing values
        df.fillna({"Check-In Time": df['Check-In Time'].mode()[0] if not df['Check-In Time'].mode().empty else "", 'Check-Out Time': df['Check-Out Time'].mode()[0] if not df['Check-Out Time'].mode().empty else "", "Amenities": df["Amenities"].mode()[0] if not df["Amenities"].mode().empty else "", "Ratings Breakdown": df["Ratings Breakdown"].mode()[0] if "Ratings Breakdown" in df and not df["Ratings Breakdown"].mode().empty else ""}, inplace=True)
        
        # Set up the vectorizer
        df['Name'] = df['Name'].astype(str)
        df['Amenities'] = df['Amenities'].astype(str)
        df['city'] = df['city'].astype(str)
        
        if 'Ratings Breakdown' in df:
            df['Ratings Breakdown'] = df['Ratings Breakdown'].astype(str)
            df['Combined'] = df['Name'] + " " + df['Amenities'] + " " + df["Ratings Breakdown"]
        else:
            df['Combined'] = df['Name'] + " " + df['Amenities']
        
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['Combined'])
        
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        abort(500, description="Failed to load hotel data")

# Manually load the data before the app starts
load_data()

# Routes
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Welcome to the Egypt Hotels Recommendation API"})

@app.route("/cities", methods=["GET"])
def get_cities():
    """Get a list of available cities"""
    cities = df['city'].unique().tolist()
    return jsonify(cities)

@app.route("/hotels/<city>", methods=["GET"])
def get_hotels_by_city(city):
    """Get all hotels in a specified city"""
    city_hotels = df[df['city'].str.lower() == city.lower()]
    
    if city_hotels.empty:
        abort(404, description=f"No hotels found in {city}")
    
    hotels = []
    for _, hotel in city_hotels.iterrows():
        hotels.append({
            "name": hotel['Name'],
            "total_rate": float(hotel['Total Rate']),
            "ratings": hotel.get('Ratings Breakdown', ""),
            "amenities": hotel['Amenities'],
            "city": hotel['city']
        })
    
    return jsonify(hotels)

@app.route("/recommend", methods=["GET"])
def recommend_hotels():
    """Recommend hotels based on city and budget"""
    city = request.args.get("city")
    budget = float(request.args.get("budget"))
    top_n = int(request.args.get("top_n", 5))
    
    if not city:
        abort(400, description="City is required")
    if not budget:
        abort(400, description="Budget is required")
    
    city_hotels = df[df['city'].str.lower() == city.lower()].copy()
    
    if city_hotels.empty:
        abort(404, description=f"No hotels found in {city}")
    
    city_hotels['Rate_Difference'] = abs(city_hotels['Total Rate'] - budget)
    closest_hotels = city_hotels.nsmallest(top_n, 'Rate_Difference')
    
    recommendations = []
    for _, hotel in closest_hotels.iterrows():
        recommendations.append({
            "name": hotel['Name'],
            "total_rate": float(hotel['Total Rate']),
            "ratings": hotel.get('Ratings Breakdown', ""),
            "amenities": hotel['Amenities'],
            "city": hotel['city']
        })
    
    return jsonify(recommendations)

@app.route("/search", methods=["POST"])
def search_hotels():
    """Search hotels based on text query"""
    data = request.get_json()
    query = data.get("query")
    city = data.get("city")
    top_n = int(data.get("top_n", 5))
    
    if not query:
        abort(400, description="Query is required")
    
    # Convert query to TF-IDF vector
    query_vector = vectorizer.transform([query])
    
    # Calculate similarity
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Filter by city if specified
    if city:
        city_mask = df['city'].str.lower() == city.lower()
        city_indices = np.where(city_mask)[0]
        filtered_scores = np.zeros(len(similarity_scores))
        filtered_scores[city_indices] = similarity_scores[city_indices]
        similarity_scores = filtered_scores
    
    # Get top results
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        if similarity_scores[idx] > 0:  # Ensure there's a match
            hotel = df.iloc[idx]
            results.append({
                "name": hotel['Name'],
                "total_rate": float(hotel['Total Rate']),
                "ratings": hotel.get('Ratings Breakdown', ""),
                "amenities": hotel['Amenities'],
                "city": hotel['city']
            })
    
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, port=5001)

