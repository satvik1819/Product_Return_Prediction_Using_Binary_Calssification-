import gradio as gr
import pandas as pd
import dill


model = dill.load(open('artifacts/model.pkl', 'rb'))

# ---------- Prediction Function ----------
def predict_return(
    product_category, product_price, order_quantity, user_age, user_gender, user_location,
    payment_method, shipping_method, discount_applied, delivery_days, past_orders,
    past_returns, past_return_rate, region
):
    try:
        # 🧠 Prepare input data for the model
        data = pd.DataFrame([{
            'Product_Category': product_category,
            'Product_Price': float(product_price),
            'Order_Quantity': int(order_quantity),
            'User_Age': int(user_age),
            'User_Gender': user_gender,
            'User_Location': user_location,
            'Payment_Method': payment_method,
            'Shipping_Method': shipping_method,
            'Discount_Applied': float(discount_applied),
            'Delivery_Days': int(delivery_days),
            'Past_Orders': int(past_orders),
            'Past_Returns': int(past_returns),
            'Past_Return_Rate': float(past_return_rate),
            'Region': region
        }])

        # ✅ Make prediction
        prediction = model.predict(data)[0]

        # ✅ Output message
        if prediction.lower() == "yes":
            return "✅ The product will be returned."
        else:
            return "🚫 The product will NOT get returned."

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ---------- Gradio UI ----------
inputs = [
    gr.Dropdown(["Electronics", "Clothing", "Home", "Beauty", "Sports"], label="Product Category"),
    gr.Number(label="Product Price"),
    gr.Number(label="Order Quantity"),
    gr.Number(label="User Age"),
    gr.Dropdown(["Male", "Female"], label="User Gender"),
    gr.Textbox(label="User Location"),
    gr.Dropdown(["Credit Card", "Debit Card", "UPI", "COD"], label="Payment Method"),
    gr.Dropdown(["Standard", "Express"], label="Shipping Method"),
    gr.Number(label="Discount Applied"),
    gr.Number(label="Delivery Days"),
    gr.Number(label="Past Orders"),
    gr.Number(label="Past Returns"),
    gr.Number(label="Past Return Rate"),
    gr.Dropdown(["North", "South", "East", "West"], label="Region")
]

output = gr.Textbox(label="Prediction")

# ✅ Launch Gradio
gr.Interface(
    fn=predict_return,
    inputs=inputs,
    outputs=output,
    title="🛍️ Product Return Prediction",
    description="Predict whether a product will be returned based on order and customer details."
).launch(share=True)   # 'share=True' gives you a temporary public link
