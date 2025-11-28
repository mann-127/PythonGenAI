import streamlit as st
import pandas as pd

# 1
# # Setup
# st.title("My First Streamlit App")
# st.write("Hello, World!")

# 2
# # Headers
# st.title("This is a title") 
# st.header("This is a header")
# st.subheader("This is a subheader")
# # Text
# st.text("Fixed width text")
# st.write("Standard text using write")
# st.markdown("**Bold** and *italic* text")
# # Code
# st.code("print('Hello')", language="python")

# 3
# # Text input
# name = st.text_input("Enter your name")
# st.write(f"Hello, {name}!")
# # Number input
# age = st.number_input("Enter your age", min_value=0, max_value=120)
# # Slider
# value = st.slider("Select a value", 0, 100, 50)
# # Select box
# option = st.selectbox( "Choose an option", ["Option 1", "Option 2", "Option 3"] )
# # Checkbox
# agree = st.checkbox("I agree")
# if agree: st.write("Thank you!")
# # Button
# if st.button("Click me"): st.write("Button clicked!")

# 4
# # Create sample data
# df = pd.DataFrame({ 'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35], 'City': ['New York', 'London', 'Paris'] })
# # Display dataframe
# st.dataframe(df)
# # Display static table
# st.table(df)
# # Display metrics
# st.metric(label="Temperature", value="70°F", delta="1.2°F")

# 5
# # Create columns
# col1, col2, col3 = st.columns(3)
# with col1: st.header("Column 1")
# st.write("Content in column 1")
# with col2: st.header("Column 2")
# st.write("Content in column 2")
# with col3: st.header("Column 3")
# st.write("Content in column 3")
# # Expander
# with st.expander("Click to expand"): st.write("Hidden content here")
# # Sidebar
# st.sidebar.title("Sidebar")
# st.sidebar.selectbox("Choose", ["A", "B", "C"])

# 6
# # Initialize session state
# if 'count' not in st.session_state: st.session_state.count = 0
# # Increment counter
# if st.button("Increment"): st.session_state.count += 1
# st.write(f"Count: {st.session_state.count}")

# 7
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.write(df)

