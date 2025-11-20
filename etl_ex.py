import pandas as pd
import sqlite3
from datetime import datetime
import numpy as np
 
# ============================================
# STEP 1: EXTRACT - Read data from CSV
# ============================================
 
def extract_data(file_path):
    """
    Extract data from CSV file
    """
    print(f"[EXTRACT] Reading data from {file_path}...")
    
    # Sample data creation (for demo purposes)
    # In real scenarios, you'd read from an actual CSV file
    sample_data = {
        'customer_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'name': ['John Doe', 'jane smith', 'BOB JONES', None, 'Alice Brown', 'Charlie', 'DAVID LEE', 'Emma Wilson'],
        'email': ['john@email.com', 'JANE@EMAIL.COM', 'bob@email', 'alice@email.com', None, 'charlie@email.com', 'david@email.com', 'emma@email.com'],
        'age': [25, 30, -5, 45, 28, 150, 35, 29],
        'purchase_amount': [100.50, 250.75, 89.99, None, 175.25, 500.00, 75.50, 320.00],
        'purchase_date': ['2024-01-15', '2024-01-16', '2024/01/17', '2024-01-18', '2024-01-19', 'invalid', '2024-01-21', '2024-01-22']
    }
    
    df = pd.DataFrame(sample_data)
    print(f"[EXTRACT] Extracted {len(df)} records")
    print("\n--- Raw Data ---")
    print(df)
    return df
 
# ============================================
# STEP 2: TRANSFORM - Clean and process data
# ============================================
 
def transform_data(df):
    """
    Clean and transform the extracted data
    """
    print("\n[TRANSFORM] Starting data transformation...")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # 1. Handle missing values
    print("  - Handling missing values...")
    df_clean['name'].fillna('Unknown', inplace=True)
    df_clean['email'].fillna('no-email@unknown.com', inplace=True)
    df_clean['purchase_amount'].fillna(df_clean['purchase_amount'].median(), inplace=True)
    
    # 2. Standardize name format (Title Case)
    print("  - Standardizing name format...")
    df_clean['name'] = df_clean['name'].str.title()
    
    # 3. Standardize email format (lowercase)
    print("  - Standardizing email format...")
    df_clean['email'] = df_clean['email'].str.lower()
    
    # 4. Validate and clean email addresses
    print("  - Validating email addresses...")
    email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    df_clean['email_valid'] = df_clean['email'].str.match(email_pattern)
    df_clean.loc[~df_clean['email_valid'], 'email'] = 'invalid@unknown.com'
    
    # 5. Clean age data (remove outliers)
    print("  - Cleaning age data...")
    df_clean.loc[(df_clean['age'] < 0) | (df_clean['age'] > 120), 'age'] = df_clean['age'].median()
    
    # 6. Standardize date format
    print("  - Standardizing dates...")
    df_clean['purchase_date'] = pd.to_datetime(df_clean['purchase_date'], errors='coerce')
    df_clean['purchase_date'].fillna(pd.Timestamp('2024-01-01'), inplace=True)
    
    # 7. Create derived columns
    print("  - Creating derived columns...")
    df_clean['age_group'] = pd.cut(df_clean['age'],
                                     bins=[0, 25, 35, 50, 100],
                                     labels=['18-25', '26-35', '36-50', '50+'])
    
    df_clean['purchase_category'] = pd.cut(df_clean['purchase_amount'],
                                            bins=[0, 100, 300, 1000],
                                            labels=['Low', 'Medium', 'High'])
    
    # 8. Add processing metadata
    df_clean['processed_at'] = datetime.now()
    
    # Drop temporary columns
    df_clean.drop('email_valid', axis=1, inplace=True)
    
    print(f"[TRANSFORM] Transformation complete. {len(df_clean)} records ready")
    print("\n--- Cleaned Data ---")
    print(df_clean)
    
    return df_clean
 
# ============================================
# STEP 3: LOAD - Write data to SQLite
# ============================================
 
def load_data(df, db_name='etl_demo.db', table_name='customers'):
    """
    Load cleaned data into SQLite database
    """
    print(f"\n[LOAD] Loading data into {db_name}...")
    
    # Connect to SQLite database (creates if doesn't exist)
    conn = sqlite3.connect(db_name)
    
    try:
        # Write DataFrame to SQLite table
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"[LOAD] Successfully loaded {len(df)} records to '{table_name}' table")
        
        # Verify the load
        print("\n[VERIFY] Reading back from database...")
        query = f"SELECT * FROM {table_name} LIMIT 5"
        result = pd.read_sql_query(query, conn)
        print(result)
        
        # Show table statistics
        count_query = f"SELECT COUNT(*) as total_records FROM {table_name}"
        total = pd.read_sql_query(count_query, conn)
        print(f"\n[STATS] Total records in database: {total['total_records'][0]}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        raise
    finally:
        conn.close()
        print(f"[LOAD] Database connection closed")
 
# ============================================
# MAIN ETL PIPELINE
# ============================================
 
def run_etl_pipeline():
    """
    Execute the complete ETL pipeline
    """
    print("="*60)
    print("ETL PIPELINE STARTED")
    print("="*60)
    
    try:
        # Extract
        raw_data = extract_data('customer_data.csv')
        
        # Transform
        cleaned_data = transform_data(raw_data)
        
        # Load
        load_data(cleaned_data)
        
        print("\n" + "="*60)
        print("ETL PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\n[FATAL ERROR] ETL Pipeline failed: {e}")
        raise
 
# Run the pipeline
if __name__ == "__main__":
    run_etl_pipeline()
