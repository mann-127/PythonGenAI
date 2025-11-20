import pandas as pd
import json
 
# ============================================
# HELPER FUNCTIONS
# ============================================
 
def create_employee_prompt(row):
    prompt = f"""Analyze this employee profile:
- Name: {row['Name']}
- Age: {row['Age']}
- Location: {row['City']}
- Department: {row['Department']}
- Annual Salary: ${row['Salary']:,.2f}
 
Provide insights on career development opportunities."""
    return prompt
 
 
def create_analysis_prompt(summary_df):
    prompt = "Analyze the following department salary data:\n\n"
    for dept in summary_df.index:
        prompt += f"{dept}:\n"
        prompt += f"  - Average Salary: ${summary_df.loc[dept, ('Salary', 'mean')]:,.2f}\n"
        prompt += f"  - Salary Range: ${summary_df.loc[dept, ('Salary', 'min')]:,.2f} - ${summary_df.loc[dept, ('Salary', 'max')]:,.2f}\n"
        prompt += f"  - Employee Count: {int(summary_df.loc[dept, ('Salary', 'count')])}\n"
        prompt += f"  - Average Age: {summary_df.loc[dept, ('Age', 'mean')]:.1f} years\n\n"
    prompt += "Provide recommendations for salary equity and workforce planning."
    return prompt
 
 
def create_api_payload(df):
    payload = {
        "model": "claude-sonnet-4-5-20250929",
        "messages": [
            {
                "role": "user",
                "content": f"""Analyze this employee dataset and provide insights:
 
{df.to_string(index=False)}
 
Please provide:
1. Key trends in the data
2. Salary distribution analysis
3. Department-wise recommendations"""
            }
        ],
        "max_tokens": 1024
    }
    return payload
 
 
def create_training_examples(df):
    training_data = []
    for _, row in df.iterrows():
        example = {
            "prompt": f"Employee in {row['Department']} department, age {row['Age']}, located in {row['City']}",
            "completion": f"Expected salary range: ${row['Salary'] * 0.9:,.0f} - ${row['Salary'] * 1.1:,.0f}"
        }
        training_data.append(example)
    return training_data
 
 
def create_personalized_prompts(df):
    """Generate personalized AI prompts for each employee"""
    prompts_list = []
    
    for _, employee in df.iterrows():
        # Calculate percentile
        salary_percentile = (df['Salary'] < employee['Salary']).sum() / len(df) * 100
        
        prompt = f"""Generate a personalized career development plan:
 
Employee Profile:
- {employee['Name']}, {employee['Age']} years old
- Current Role: {employee['Department']}
- Location: {employee['City']}
- Current Salary: ${employee['Salary']:,.2f} (Top {100-salary_percentile:.0f}%)
 
Consider their experience level and market position to suggest:
1. Skill development opportunities
2. Potential career progression paths
3. Competitive salary expectations for next role"""
        
        prompts_list.append({
            'employee': employee['Name'],
            'prompt': prompt
        })
    
    return prompts_list
 
 
# ============================================
# MAIN FUNCTION
# ============================================
 
def main():
    # ============================================
    # 1. BASIC CSV READING WITH PANDAS
    # ============================================
    
    # Read CSV file
    df = pd.read_csv('sample_data.csv')
    
    # Display basic info
    print("Dataset Overview:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nBasic Statistics:")
    print(df.describe())
    
    
    # ============================================
    # 2. DATA CLEANING FOR GEN AI
    # ============================================
    
    # Remove missing values
    df_clean = df.dropna()
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Convert data types
    df_clean['Age'] = df_clean['Age'].astype(int)
    df_clean['Salary'] = df_clean['Salary'].astype(float)
    
    
    # ============================================
    # 3. PREPARE DATA FOR GEN AI PROMPTS
    # ============================================
    
    # Generate prompts for each employee
    prompts = df.apply(create_employee_prompt, axis=1)
    print("\n\nSample Prompt for GenAI:")
    print(prompts.iloc[0])
    
    
    # ============================================
    # 4. AGGREGATE DATA FOR AI ANALYSIS
    # ============================================
    
    # Group by department
    dept_summary = df.groupby('Department').agg({
        'Salary': ['mean', 'min', 'max', 'count'],
        'Age': 'mean'
    }).round(2)
    
    print("\n\nDepartment Summary:")
    print(dept_summary)
    
    # Create prompt from aggregated data
    analysis_prompt = create_analysis_prompt(dept_summary)
    print("\n\nAggregated Analysis Prompt:")
    print(analysis_prompt)
    
    
    # ============================================
    # 5. CONVERT TO JSON FOR API CALLS
    # ============================================
    
    # Convert DataFrame to JSON for API payload
    json_data = df.to_json(orient='records', indent=2)
    print("\n\nJSON Format (for API calls):")
    print(json_data[:500] + "...")
    
    # Create structured payload for GenAI API
    api_payload = create_api_payload(df)
    print("\n\nAPI Payload Structure:")
    print(json.dumps(api_payload, indent=2)[:600] + "...")
    
    
    # ============================================
    # 6. CREATE TRAINING DATA FORMAT
    # ============================================
    
    # Format for fine-tuning or few-shot learning
    training_examples = create_training_examples(df)
    print("\n\nTraining Data Examples:")
    print(json.dumps(training_examples[:2], indent=2))
    
    
    # ============================================
    # 7. FILTER AND CONTEXTUALIZE DATA
    # ============================================
    
    # Filter high earners for specific analysis
    high_earners = df[df['Salary'] > 80000]
    
    context_prompt = f"""Context: We have {len(high_earners)} high-earning employees (>$80k) out of {len(df)} total employees.
 
High Earners:
{high_earners.to_string(index=False)}
 
Generate a retention strategy for these key employees."""
    
    print("\n\nContextualized Prompt:")
    print(context_prompt)
    
    
    # ============================================
    # 8. CREATE BATCH PROCESSING DATA
    # ============================================
    
    # Split data for batch processing
    batch_size = 3
    batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    print(f"\n\nCreated {len(batches)} batches for processing")
    print(f"First batch:\n{batches[0]}")
    
    
    # ============================================
    # 9. EXPORT PROCESSED DATA
    # ============================================
    
    # Save cleaned data
    df_clean.to_csv('cleaned_data.csv', index=False)
    
    # Save as JSON for GenAI
    with open('genai_input.json', 'w') as f:
        json.dump(training_examples, f, indent=2)
    
    print("\n\nFiles saved:")
    print("- cleaned_data.csv")
    print("- genai_input.json")
    
    
    # ============================================
    # 10. REAL-WORLD GENAI USE CASE
    # ============================================
    
    personalized = create_personalized_prompts(df)
    print("\n\nPersonalized GenAI Prompt Example:")
    print(personalized[0]['prompt'])
 
 
# ============================================
# ENTRY POINT
# ============================================
 
if __name__ == "__main__":
    main()
