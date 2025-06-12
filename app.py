import streamlit as st

# Initialize session state for resetting inputs
if 'reset' not in st.session_state:
    st.session_state.reset = False

def calculator(num1, num2, operator):
    if operator == "+":
        return num1 + num2
    elif operator == "-":
        return num1 - num2
    elif operator == "*":
        return num1 * num2
    elif operator == "/":
        if num2 == 0:
            return "Error: Cannot divide by zero."
        else:
            return num1 / num2
    else:
        return "Invalid operator. Please select one of: +, -, *, /"

# App title
st.title("Simple Calculator")

# Input fields
num1 = st.number_input("Enter the first number:", value=0.0, step=0.1, key="num1" if not st.session_state.reset else "num1_reset")
num2 = st.number_input("Enter the second number:", value=0.0, step=0.1, key="num2" if not st.session_state.reset else "num2_reset")
operator = st.selectbox("Select the operator:", ["+", "-", "*", "/"], key="operator" if not st.session_state.reset else "operator_reset")

# Calculate button
if st.button("Calculate"):
    try:
        result = calculator(num1, num2, operator)
        if isinstance(result, str):
            st.error(result)
        else:
            st.write(f"The result is: {result}")
    except ValueError:
        st.error("Error: Please enter valid numbers.")

# Reset button for another calculation
if st.button("Clear for Another Calculation"):
    st.session_state.reset = True
    st.experimental_rerun()

# Reset the reset flag after rerun
if st.session_state.reset:
    st.session_state.reset = False
