import requests
import unittest
from unittest.mock import patch, Mock
import sqlite3
import os
import streamlit_login_app as app

# # --- 1. Basic Math Test ---
# class TestMath(unittest.TestCase):
#     def test_addition(self):
#         self.assertEqual(2 + 3, 5)
 
#     def test_subtraction(self):
#         self.assertEqual(10 - 5, 5)
 
 
# # --- 2. String Methods ---
# class TestStringMethods(unittest.TestCase):
#     def test_upper(self):
#         self.assertEqual("hello".upper(), "HELLO")
 
#     def test_isupper(self):
#         self.assertTrue("HELLO".isupper())
#         self.assertFalse("Hello".isupper())
 
 
# # --- 3. List Operations ---
# class TestListOperations(unittest.TestCase):
#     def test_append(self):
#         nums = [1, 2]
#         nums.append(3)
#         self.assertEqual(nums, [1, 2, 3])
 
#     def test_remove(self):
#         nums = [1, 2, 3]
#         nums.remove(2)
#         self.assertNotIn(2, nums)
 
 
# # --- 4. Exception Handling ---
# def divide(a, b):
#     if b == 0:
#         raise ValueError("Cannot divide by zero")
#     return a / b
 
 
# class TestExceptions(unittest.TestCase):
#     def test_divide_by_zero(self):
#         with self.assertRaises(ValueError):
#             divide(10, 0)
 
 
# # --- 5. Setup and Teardown Example ---
# class TestSetupTeardown(unittest.TestCase):
#     def setUp(self):
#         self.data = [1, 2, 3]
 
#     def tearDown(self):
#         self.data.clear()
 
#     def test_sum(self):
#         self.assertEqual(sum(self.data), 6)
 
 
# # --- 6. Testing a Custom Class ---
# class Calculator:
#     def multiply(self, a, b):
#         return a * b
 
#     def divide(self, a, b):
#         if b == 0:
#             raise ZeroDivisionError
#         return a / b
 
 
# class TestCalculator(unittest.TestCase):
#     def setUp(self):
#         self.calc = Calculator()
 
#     def test_multiply(self):
#         self.assertEqual(self.calc.multiply(3, 4), 12)
 
#     def test_divide(self):
#         self.assertAlmostEqual(self.calc.divide(10, 2), 5.0)
 
 
# # --- 7. Using subTest() for Multiple Cases ---
# class TestParameterized(unittest.TestCase):
#     def test_add_cases(self):
#         cases = [(1, 1, 2), (2, 2, 4), (3, 5, 8)]
#         for a, b, expected in cases:
#             with self.subTest(a=a, b=b):
#                 self.assertEqual(a + b, expected)
 
 
# # --- 8. Mocking Example ---
# class PaymentProcessor:
#     def __init__(self, service):
#         self.service = service
 
#     def pay(self, amount):
#         response = self.service.charge(amount)
#         return response["status"] == "success"
 
 
# class TestPaymentProcessor(unittest.TestCase):
#     @patch("__main__.PaymentProcessor")
#     def test_payment_success(self, MockProcessor):
#         instance = MockProcessor.return_value
#         instance.pay.return_value = True
#         self.assertTrue(instance.pay(100))
 
 
# # --- 9. Testing Edge Cases ---
# def factorial(n):
#     if n < 0:
#         raise ValueError("Negative not allowed")
#     return 1 if n == 0 else n * factorial(n - 1)
 
 
# class TestFactorial(unittest.TestCase):
#     def test_factorial_zero(self):
#         self.assertEqual(factorial(0), 1)
 
#     def test_factorial_positive(self):
#         self.assertEqual(factorial(5), 120)
 
#     def test_factorial_negative(self):
#         with self.assertRaises(ValueError):
#             factorial(-1)
 
 
# # --- 10. File Operations ---
# def read_file_content(filename):
#     with open(filename, "r") as f:
#         return f.read().strip()
 
 
# class TestFileOperations(unittest.TestCase):
#     def test_file_read(self):
#         with open("temp.txt", "w") as f:
#             f.write("Hello World")
#         content = read_file_content("temp.txt")
#         self.assertEqual(content, "Hello World")


# # 14. Mocking API Call
# def get_weather(city):
#     response = requests.get(f"http://api.weather/{city}")
#     return response.json()
 
 
# class TestAPICall(unittest.TestCase):
#     @patch("builtins.print")
#     @patch("requests.get")
#     def test_weather_api_mock(self, mock_get, mock_print):
#         mock_get.return_value.json.return_value = {"temp": 30}
#         result = get_weather("Delhi")
#         self.assertEqual(result["temp"], 30)

 
# if __name__ == "__main__":
#     unittest.main()


# # test_database_connection.py
 
# DB_PATH = "etl_ai_demo.db"
# TABLE_NAME = "products"
 
# class TestProductsTable(unittest.TestCase):
 
#     @classmethod
#     def setUpClass(cls):
#         """Connect to the SQLite database"""
#         cls.conn = sqlite3.connect(DB_PATH)
#         cls.cursor = cls.conn.cursor()
 
#     def test_db_connection(self):
#         """Verify database connection can be established"""
#         conn = sqlite3.connect(DB_PATH)
#         self.assertIsNotNone(conn)
#         conn.close()
 
#     def test_table_exists(self):
#         """Check if 'products' table exists"""
#         self.cursor.execute("""
#             SELECT name FROM sqlite_master WHERE type='table' AND name=?;
#         """, (TABLE_NAME,))
#         table = self.cursor.fetchone()
#         self.assertIsNotNone(table, f"Table '{TABLE_NAME}' does not exist")
 
#     def test_columns_exist(self):
#         """Check required columns in 'products' table"""
#         expected_columns = [
#             "product_id", "product_name", "category", "price", "description",
#             "price_tier", "processed_at", "ai_tags", "ai_marketing_summary", "target_audience"
#         ]
#         self.cursor.execute(f"PRAGMA table_info({TABLE_NAME});")
#         columns = [col[1] for col in self.cursor.fetchall()]
#         for col in expected_columns:
#             self.assertIn(col, columns, f"Column '{col}' missing in table")
 
#     def test_row_count(self):
#         """Check that the table has at least 1 row"""
#         self.cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
#         count = self.cursor.fetchone()[0]
#         self.assertGreater(count, 0, "Table has no rows")
 
#     def test_sample_data(self):
#         """Verify specific sample data exists"""
#         self.cursor.execute(f"SELECT * FROM {TABLE_NAME} WHERE product_name = 'Laptop';")
#         row = self.cursor.fetchone()
#         self.assertIsNotNone(row, "Sample product 'Laptop' not found")
#         self.assertEqual(row[2], "Electronics")  # category
#         self.assertAlmostEqual(row[3], 999.99)   # price
 
#     @classmethod
#     def tearDownClass(cls):
#         """Close DB connection"""
#         cls.conn.close()
 
# if __name__ == "__main__":
#     unittest.main(verbosity=2)


# test_streamlit_login_app.py
 
class TestStreamlitLogin(unittest.TestCase):
 
    # ---- Logic Tests ----
    def test_correct_login(self):
        self.assertTrue(app.authenticate_user("admin", "12345"))
 
    def test_incorrect_login(self):
        self.assertFalse(app.authenticate_user("user", "wrongpass"))
 
    def test_empty_login_fields(self):
        self.assertFalse(app.authenticate_user("", ""))
 
    def test_password_reset_success(self):
        self.assertTrue(app.reset_password("user", "newpass123"))
 
    def test_password_reset_failure(self):
        self.assertFalse(app.reset_password("", ""))
 
    # ---- UI Tests ----
    @patch("streamlit_login_app.st")
    def test_ui_login_success(self, mock_st):
        mock_st.sidebar.selectbox.return_value = "Login"
        mock_st.text_input.side_effect = ["admin", "12345"]
        mock_st.button.return_value = True
 
        app.main()
        mock_st.success.assert_called_with("✅ Login successful!")
 
    @patch("streamlit_login_app.st")
    def test_ui_login_failure(self, mock_st):
        mock_st.sidebar.selectbox.return_value = "Login"
        mock_st.text_input.side_effect = ["user", "wrongpass"]
        mock_st.button.return_value = True
 
        app.main()
        mock_st.error.assert_called_with("❌ Invalid credentials")
 
    @patch("streamlit_login_app.st")
    def test_ui_forgot_password_success(self, mock_st):
        mock_st.sidebar.selectbox.return_value = "Forgot Password"
        mock_st.text_input.side_effect = ["user", "newpassword"]
        mock_st.button.return_value = True
 
        app.main()
        mock_st.success.assert_called_with("✅ Password reset successful!")
 
    @patch("streamlit_login_app.st")
    def test_ui_forgot_password_failure(self, mock_st):
        mock_st.sidebar.selectbox.return_value = "Forgot Password"
        mock_st.text_input.side_effect = ["", ""]
        mock_st.button.return_value = True
 
        app.main()
        mock_st.error.assert_called_with("❌ Please fill in all fields")
 
if __name__ == "__main__":
    unittest.main()
