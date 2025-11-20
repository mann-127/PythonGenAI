import pandas as pd
import sqlite3
from datetime import datetime
import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL")
# ============================================
# Configuration
# ============================================
 
# Free API Options (choose one):
# 1. Groq (recommended - very fast, free tier)
# 2. Hugging Face Inference API
# 3. OpenRouter (various free models)
 
# Alternative: Hugging Face (if you prefer)
# HF_API_KEY = "your_hf_token_here"  # Get at: https://huggingface.co/settings/tokens
# HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
 
# ============================================
# EXTRACT
# ============================================
 
def extract_data():
    """Extract data from source"""
    print(f"[EXTRACT] Reading data...")
    
    sample_data = {
        'product_id': [1, 2, 3, 4, 5],
        'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
        'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Audio'],
        'price': [999.99, 25.50, 75.00, 350.00, 150.00],
        'description': [
            'High performance laptop',
            'Wireless mouse',
            'Mechanical keyboard',
            '27 inch monitor',
            'Noise cancelling headphones'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    print(f"[EXTRACT] Extracted {len(df)} records\n")
    print(df)
    return df
 
# ============================================
# AI ENRICHMENT (using Free GPT Model)
# ============================================
 
def enrich_with_ai_groq(product_name, description, category):
    """
    Send data to Groq's free LLM API for enrichment
    Model: llama-3.1-8b-instant (fast and free)
    """
    
    if GROQ_API_KEY:
        print("  [AI] Skipping AI enrichment - API key not configured")
        return {
            'ai_tags': 'premium,quality',
            'ai_summary': f'Quality {category.lower()} product',
            'target_audience': 'General consumers'
        }
    
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""Analyze this product and provide enrichment data in JSON format:
Product: {product_name}
Category: {category}
Description: {description}
 
Provide exactly this JSON structure (no other text):
{{
  "ai_tags": "tag1,tag2,tag3",
  "ai_summary": "brief marketing summary (max 50 words)",
  "target_audience": "who should buy this"
}}"""
 
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7, # max = 2.0, least = 0.1
            "max_tokens": 200
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result['choices'][0]['message']['content']
            
            # Extract JSON from response
            start = ai_response.find('{')
            end = ai_response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = ai_response[start:end]
                enrichment = json.loads(json_str)
                return enrichment
            else:
                raise ValueError("No valid JSON in response")
        else:
            print(f"  [AI] API Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"  [AI] Enrichment failed: {e}")
        return None
 
def enrich_with_ai_huggingface(product_name, description, category):
    """
    Alternative: Use Hugging Face Inference API (free)
    """
    # Placeholder for HF implementation
    # Similar structure to Groq but uses HF endpoints
    pass
 
# ============================================
# TRANSFORM
# ============================================
 
def transform_data(df):
    """Transform and enrich data with AI"""
    print(f"\n[TRANSFORM] Starting transformation...")
    
    df_clean = df.copy()
    
    # Basic transformations
    df_clean['price_tier'] = pd.cut(df_clean['price'],
                                     bins=[0, 50, 200, 1000],
                                     labels=['Budget', 'Mid-Range', 'Premium'])
    
    df_clean['processed_at'] = datetime.now()
    
    # AI Enrichment
    print("\n[AI ENRICHMENT] Calling free GPT model for each product...")
    
    ai_tags = []
    ai_summaries = []
    target_audiences = []
    
    for idx, row in df_clean.iterrows():
        print(f"  Processing product {idx+1}/{len(df_clean)}: {row['product_name']}...")
        
        enrichment = enrich_with_ai_groq(
            row['product_name'],
            row['description'],
            row['category']
        )
        
        if enrichment:
            ai_tags.append(enrichment.get('ai_tags', 'N/A'))
            ai_summaries.append(enrichment.get('ai_summary', 'N/A'))
            target_audiences.append(enrichment.get('target_audience', 'N/A'))
        else:
            ai_tags.append('N/A')
            ai_summaries.append('N/A')
            target_audiences.append('N/A')
        
        # Rate limiting (be nice to free APIs)
        time.sleep(0.5)
    
    df_clean['ai_tags'] = ai_tags
    df_clean['ai_marketing_summary'] = ai_summaries
    df_clean['target_audience'] = target_audiences
    
    print(f"\n[TRANSFORM] Transformation complete\n")
    print(df_clean[['product_name', 'ai_tags', 'target_audience']])
    
    return df_clean
 
# ============================================
# LOAD
# ============================================
 
def load_data(df, db_name='etl_ai_demo.db', table_name='products'):
    """Load data into SQLite"""
    print(f"\n[LOAD] Loading data into {db_name}...")
    
    conn = sqlite3.connect(db_name)
    
    try:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"[LOAD] Successfully loaded {len(df)} records")
        
        # Show loaded data
        query = f"SELECT product_name, category, price, ai_tags, target_audience FROM {table_name}"
        result = pd.read_sql_query(query, conn)
        print("\n--- Data in Database ---")
        print(result)
        
    finally:
        conn.close()
 
# ============================================
# UPDATE EXAMPLE - Incremental ETL
# ============================================
 
def extract_new_data():
    """Simulate new data arriving"""
    print(f"\n[UPDATE] New data arrived...")
    
    new_data = {
        'product_id': [6, 7, 2],  # Note: product_id 2 already exists (update)
        'product_name': ['Webcam', 'USB Cable', 'Gaming Mouse'],  # Mouse updated
        'category': ['Accessories', 'Accessories', 'Accessories'],
        'price': [89.99, 12.50, 45.00],  # Mouse price changed
        'description': [
            '1080p webcam',
            'USB-C cable 2m',
            'RGB gaming mouse with 16000 DPI'  # Updated description
        ]
    }
    
    df = pd.DataFrame(new_data)
    print(df)
    return df
 
def update_data_in_db(df_new, db_name='etl_ai_demo.db', table_name='products'):
    """
    Perform incremental update:
    - INSERT new records
    - UPDATE existing records
    """
    print(f"\n[UPDATE] Processing incremental update...")
    
    conn = sqlite3.connect(db_name)
    
    try:
        # Read existing data
        existing_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        existing_ids = set(existing_df['product_id'])
        
        # Separate new vs existing
        new_records = df_new[~df_new['product_id'].isin(existing_ids)]
        update_records = df_new[df_new['product_id'].isin(existing_ids)]
        
        print(f"  - New records to INSERT: {len(new_records)}")
        print(f"  - Existing records to UPDATE: {len(update_records)}")
        
        # Transform new data
        if len(new_records) > 0:
            new_records_transformed = transform_data(new_records)
            new_records_transformed.to_sql(table_name, conn, if_exists='append', index=False)
            print(f"[UPDATE] Inserted {len(new_records)} new records")
        
        # Update existing records
        for idx, row in update_records.iterrows():
            print(f"\n  Updating product_id={row['product_id']}...")
            
            # Get AI enrichment for updated record
            enrichment = enrich_with_ai_groq(
                row['product_name'],
                row['description'],
                row['category']
            )
            
            if enrichment:
                update_query = f"""
                UPDATE {table_name}
                SET product_name = ?,
                    description = ?,
                    price = ?,
                    ai_tags = ?,
                    ai_marketing_summary = ?,
                    target_audience = ?,
                    processed_at = ?
                WHERE product_id = ?
                """
                
                conn.execute(update_query, (
                    row['product_name'],
                    row['description'],
                    row['price'],
                    enrichment.get('ai_tags', 'N/A'),
                    enrichment.get('ai_summary', 'N/A'),
                    enrichment.get('target_audience', 'N/A'),
                    datetime.now(),
                    row['product_id']
                ))
                
        conn.commit()
        print(f"\n[UPDATE] Updated {len(update_records)} records")
        
        # Show final state
        print("\n--- Updated Database State ---")
        final_df = pd.read_sql_query(f"SELECT product_id, product_name, price, ai_tags FROM {table_name}", conn)
        print(final_df)
        
    finally:
        conn.close()
 
# ============================================
# MAIN PIPELINE
# ============================================
 
def run_initial_etl():
    """Run initial ETL pipeline"""
    print("="*70)
    print("INITIAL ETL PIPELINE - WITH AI ENRICHMENT")
    print("="*70)
    
    raw_data = extract_data()
    cleaned_data = transform_data(raw_data)
    load_data(cleaned_data)
    
    print("\n" + "="*70)
    print("INITIAL ETL COMPLETED")
    print("="*70)
 
def run_incremental_update():
    """Run incremental update"""
    print("\n\n" + "="*70)
    print("INCREMENTAL UPDATE PIPELINE")
    print("="*70)
    
    new_data = extract_new_data()
    update_data_in_db(new_data)
    
    print("\n" + "="*70)
    print("INCREMENTAL UPDATE COMPLETED")
    print("="*70)
 
if __name__ == "__main__":
    # Run initial load
    run_initial_etl()
    
    # Simulate incremental update after some time
    print("\n\n[SYSTEM] Waiting 3 seconds before incremental update...")
    time.sleep(3)
    
    run_incremental_update()
    
    print("\n\n[INFO] To use AI enrichment:")
    print("1. Get free API key from https://console.groq.com")
    print("2. Replace GROQ_API_KEY in the code")
    print("3. Run again to see AI-generated tags and summaries!")
