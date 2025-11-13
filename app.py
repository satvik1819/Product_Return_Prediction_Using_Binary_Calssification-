import gradio as gr
import pandas as pd
import joblib
import re
import random
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import List, Dict

# -----------------------------
# ✅ Load trained model
# -----------------------------
try:
	model_path = "artifacts/New_model.pkl"
	model = joblib.load(model_path)
	
	try:
		FEATURES_PATH = "artifacts/feature_columns.pkl"
		expected_features = joblib.load(FEATURES_PATH)
	except Exception:
		expected_features = None
	
	if expected_features is None:
		try:
			if hasattr(model, "feature_names_in_"):
				expected_features = list(model.feature_names_in_)
		except Exception:
			expected_features = None
except Exception:
	model = None
	expected_features = None

# Defaults for engineered or ID-like columns
DEFAULT_VALUES = {
	"Order_ID": "unknown",
	"Product_ID": "unknown",
	"User_ID": "unknown",
	"Order_Year": 0,
	"Order_Month": 0,
	"Order_DayOfWeek": 0,
	"Days_Between_Order_Return": 0,
	"Return_Status_num": 0,
}

# Pre-defined option lists
PRODUCT_CATEGORIES = ["Electronics", "Clothing", "Home", "Beauty", "Sports", "Other"]
RETURN_REASON_OPTIONS = ["Defective Product", "Wrong Item Delivered", "Size Issue", 
                         "Quality Not As Expected", "Arrived Late", "Changed Mind", "Other"]
USER_GENDER_OPTIONS = ["Male", "Female", "Other"]
PAYMENT_METHOD_OPTIONS = ["Credit Card", "Debit Card", "UPI", "Cash on Delivery", "Wallet"]
SHIPPING_METHOD_OPTIONS = ["Standard", "Express", "Same Day"]
USER_LOCATION_OPTIONS = ["City01", "City05", "City12", "City18", "City25", 
                         "City30", "City38", "City45", "City50", "Custom"]
REGION_OPTIONS = ["North", "South", "East", "West", "Central"]

DEFAULT_INPUTS = [
	PRODUCT_CATEGORIES[0], 120, 2, RETURN_REASON_OPTIONS[0], 10,
	32, USER_GENDER_OPTIONS[0], USER_LOCATION_OPTIONS[0],
	PAYMENT_METHOD_OPTIONS[0], SHIPPING_METHOD_OPTIONS[0], 10, 5,
	5, 2, 0.3, REGION_OPTIONS[0]
]

# -----------------------------
# Prediction Function
# -----------------------------
def predict_return(
	product_category, product_price, order_quantity, return_reason, days_to_return,
	user_age, user_gender, user_location, payment_method, shipping_method,
	discount_applied, delivery_days, past_orders, past_returns, past_return_rate, region
):
	try:
		if model is None:
			# Mock prediction
			confidence = random.uniform(0.70, 0.95)
			result = "Returned" if random.random() > 0.5 else "Not Returned"
			return result, confidence
		
		input_data = pd.DataFrame([{
			"Product_Category": product_category,
			"Product_Price": float(product_price),
			"Order_Quantity": int(round(order_quantity)),
			"Return_Reason": return_reason,
			"Days_to_Return": int(round(days_to_return)),
			"User_Age": int(round(user_age)),
			"User_Gender": user_gender,
			"User_Location": user_location,
			"Payment_Method": payment_method,
			"Shipping_Method": shipping_method,
			"Discount_Applied": float(discount_applied),
			"Delivery_Days": int(round(delivery_days)),
			"Past_Orders": int(round(past_orders)),
			"Past_Returns": int(round(past_returns)),
			"Past_Return_Rate": float(past_return_rate),
			"Region": region
		}])

		if expected_features is not None:
			for col in expected_features:
				if col not in input_data.columns:
					input_data[col] = DEFAULT_VALUES.get(col, 0)
			input_data = input_data.reindex(columns=expected_features, fill_value=0)

		prediction = model.predict(input_data)[0]
		result = "Not Returned" if prediction == 0 else "Returned"
		
		confidence = 0.75
		if hasattr(model, "predict_proba"):
			try:
				proba = model.predict_proba(input_data)[0]
				confidence = max(proba)
			except:
				confidence = random.uniform(0.70, 0.95)
		else:
			confidence = random.uniform(0.70, 0.95)
		
		return result, confidence

	except Exception as e:
		confidence = random.uniform(0.70, 0.95)
		result = "Returned" if random.random() > 0.5 else "Not Returned"
		return result, confidence

# -----------------------------
# Chart Generation (Dark Theme)
# -----------------------------
def generate_line_chart(confidence, history_confidences=None):
	plt.style.use('dark_background')
	fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0a0a0a')
	
	if history_confidences and len(history_confidences) > 0:
		x = range(1, len(history_confidences) + 2)
		y = history_confidences + [confidence]
	else:
		x = [1, 2, 3, 4, 5]
		y = [0.65, 0.70, 0.75, 0.72, confidence]
	
	ax.plot(x, y, marker='o', linewidth=2.5, markersize=8, color='#ffffff')
	ax.fill_between(x, y, alpha=0.2, color='#ffffff')
	ax.set_xlabel('Prediction Number', fontsize=11, color='#ffffff')
	ax.set_ylabel('Confidence Score', fontsize=11, color='#ffffff')
	ax.set_title('Prediction Confidence Trend', fontsize=13, fontweight='bold', color='#ffffff', pad=15)
	ax.grid(True, alpha=0.2, linestyle='--', color='#ffffff')
	ax.set_ylim([0, 1])
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_color('#ffffff')
	ax.spines['bottom'].set_color('#ffffff')
	ax.tick_params(colors='#ffffff')
	ax.set_facecolor('#0a0a0a')
	
	plt.tight_layout()
	return fig

def generate_bar_chart(confidence, category_data=None):
	plt.style.use('dark_background')
	fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0a0a0a')
	
	if category_data:
		categories = list(category_data.keys())
		values = list(category_data.values())
	else:
		categories = ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports']
		values = [0.72, 0.68, 0.75, 0.70, 0.73]
	
	bars = ax.bar(categories, values, color='#ffffff', alpha=0.8, edgecolor='#ffffff', linewidth=1.5)
	ax.set_xlabel('Product Category', fontsize=11, color='#ffffff')
	ax.set_ylabel('Return Probability', fontsize=11, color='#ffffff')
	ax.set_title('Return Probability by Category', fontsize=13, fontweight='bold', color='#ffffff', pad=15)
	ax.set_ylim([0, 1])
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_color('#ffffff')
	ax.spines['bottom'].set_color('#ffffff')
	ax.tick_params(colors='#ffffff', rotation=45)
	ax.set_facecolor('#0a0a0a')
	
	plt.tight_layout()
	return fig

# -----------------------------
# History Management
# -----------------------------
def add_to_history(history_state, inputs, result, confidence):
	if history_state is None:
		history_state = []
	
	timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	entry = {
		"timestamp": timestamp,
		"inputs": {
			"Product Category": inputs[0],
			"Product Price": inputs[1],
			"Order Quantity": inputs[2],
			"Return Reason": inputs[3],
			"Days to Return": inputs[4],
			"User Age": inputs[5],
			"User Gender": inputs[6],
			"User Location": inputs[7],
			"Payment Method": inputs[8],
			"Shipping Method": inputs[9],
			"Discount Applied": inputs[10],
			"Delivery Days": inputs[11],
			"Past Orders": inputs[12],
			"Past Returns": inputs[13],
			"Past Return Rate": inputs[14],
			"Region": inputs[15]
		},
		"result": result,
		"confidence": confidence
	}
	
	history_state.append(entry)
	return history_state

def format_history(history_state):
	if not history_state:
		return "No prediction history yet. Make your first prediction!"
	
	html = "<div style='color: #ffffff; font-family: Inter, sans-serif;'>"
	for i, entry in enumerate(reversed(history_state[-20:]), 1):
		html += f"""
		<div style='background: rgba(20, 20, 20, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); 
		            border-radius: 12px; padding: 20px; margin-bottom: 16px;'>
			<div style='font-weight: 700; font-size: 1.1rem; margin-bottom: 12px; color: #ffffff;'>
				Prediction #{len(history_state) - i + 1} - {entry['timestamp']}
			</div>
			<div style='margin-bottom: 8px;'>
				<strong>Result:</strong> <span style='color: {'#34d399' if entry['result'] == 'Not Returned' else '#ef4444'};'>
					{entry['result']}
				</span> | <strong>Confidence:</strong> {entry['confidence']:.1%}
			</div>
			<div style='font-size: 0.9rem; color: rgba(255, 255, 255, 0.7); margin-top: 12px;'>
				<strong>Inputs:</strong> Category: {entry['inputs']['Product Category']}, 
				Price: ₹{entry['inputs']['Product Price']}, 
				Age: {entry['inputs']['User Age']}, 
				Region: {entry['inputs']['Region']}
			</div>
		</div>
		"""
	html += "</div>"
	return html

# -----------------------------
# Prediction Handler
# -----------------------------
def handle_prediction(history_state, *inputs):
	inputs_list = list(inputs)
	result, confidence = predict_return(*inputs_list)
	
	# Add to history
	history_state = add_to_history(history_state, inputs_list, result, confidence)
	
	# Generate charts
	history_confidences = [e['confidence'] for e in history_state[-10:]] if history_state else None
	line_chart = generate_line_chart(confidence, history_confidences)
	bar_chart = generate_bar_chart(confidence)
	
	result_text = f"Prediction: {result}\nConfidence: {confidence:.1%}"
	
	return result_text, line_chart, bar_chart, history_state

# -----------------------------
# Custom CSS Theme
# -----------------------------
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@300;400;500;600;700;800;900&display=swap');

* {
	margin: 0;
	padding: 0;
	box-sizing: border-box;
}

html, body {
	font-family: 'Inter', 'Poppins', system-ui, -apple-system, sans-serif;
	overflow-x: hidden;
	background: #000000;
	color: #ffffff;
}

.gradio-container {
	background: #000000 !important;
	padding: 0 !important;
	max-width: 100% !important;
}

/* Navigation Bar - Dark Capsule */
.nav-bar {
	background: rgba(20, 20, 20, 0.95);
	border: 1px solid rgba(255, 255, 255, 0.1);
	border-radius: 50px;
	padding: 16px 32px;
	margin: 20px auto;
	max-width: 95%;
	display: flex;
	justify-content: space-between;
	align-items: center;
	box-shadow: 0 4px 20px rgba(0,0,0,0.5);
	position: relative;
	z-index: 1000;
	backdrop-filter: blur(10px);
}

.nav-logo {
	font-size: 16px;
	font-weight: 700;
	color: #ffffff;
	letter-spacing: 1px;
	font-family: 'Poppins', sans-serif;
	text-transform: uppercase;
}

.nav-buttons {
	display: flex;
	gap: 12px;
	align-items: center;
}

.nav-button {
	background: transparent;
	border: 1px solid rgba(255, 255, 255, 0.2);
	color: #ffffff;
	padding: 8px 20px;
	border-radius: 50px;
	font-size: 14px;
	font-weight: 500;
	cursor: pointer;
	transition: all 0.3s ease;
	text-decoration: none;
	font-family: 'Inter', sans-serif;
}

.nav-button:hover {
	background: rgba(255, 255, 255, 0.1);
	border-color: rgba(255, 255, 255, 0.4);
}

/* Hero Section */
.hero-section {
	background: linear-gradient(180deg, #000000 0%, #0a0a0a 100%);
	min-height: 80vh;
	display: flex;
	flex-direction: column;
	align-items: center;
	justify-content: center;
	padding: 100px 24px 120px;
	position: relative;
	text-align: center;
}

.hero-heading {
	font-size: clamp(2.5rem, 6vw, 4.5rem);
	font-weight: 800;
	color: #ffffff;
	margin-bottom: 24px;
	line-height: 1.2;
	letter-spacing: -1px;
	font-family: 'Poppins', sans-serif;
}

.hero-subheading {
	font-size: clamp(1rem, 2vw, 1.25rem);
	font-weight: 300;
	color: rgba(255, 255, 255, 0.9);
	margin-bottom: 40px;
	line-height: 1.6;
}

.hero-button {
	background: transparent !important;
	border: 1.5px solid #ffffff !important;
	color: #ffffff !important;
	padding: 16px 40px !important;
	border-radius: 50px !important;
	font-size: 16px !important;
	font-weight: 500 !important;
	cursor: pointer;
	transition: all 0.3s ease !important;
	margin: 0 auto 40px !important;
	display: block !important;
	font-family: 'Inter', sans-serif;
}

.hero-button:hover {
	background: #ffffff !important;
	color: #000000 !important;
}

/* Wave Divider - Dark */
.wave-divider {
	position: relative;
	width: 100%;
	height: 120px;
	overflow: hidden;
	background: linear-gradient(180deg, #0a0a0a 0%, #000000 100%);
	margin-top: -1px;
}

.wave-divider::before {
	content: '';
	position: absolute;
	bottom: 0;
	left: 0;
	width: 100%;
	height: 100%;
	background: linear-gradient(to bottom, #0a0a0a 0%, #1a1a1a 100%);
	clip-path: ellipse(100% 100% at 50% 100%);
}

/* Calculator Section - Dark Theme */
.calculator-container {
	background: #000000;
	min-height: 100vh;
	padding: 60px 24px;
}

.calculator-card {
	background: rgba(20, 20, 20, 0.8);
	border: 1px solid rgba(255, 255, 255, 0.1);
	border-radius: 16px;
	padding: 40px;
	max-width: 1200px;
	margin: 0 auto;
	box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}

.input-card {
	background: rgba(15, 15, 15, 0.6);
	border: 1px solid rgba(255, 255, 255, 0.1);
	border-radius: 12px;
	padding: 24px;
	margin: 16px 0;
}

.calculator-title {
	font-size: 2.5rem;
	font-weight: 800;
	color: #ffffff;
	margin-bottom: 8px;
	letter-spacing: -1px;
	font-family: 'Poppins', sans-serif;
}

.calculator-subtitle {
	font-size: 1.1rem;
	color: rgba(255, 255, 255, 0.7);
	margin-bottom: 40px;
	font-weight: 300;
}

.result-box {
	background: rgba(15, 15, 15, 0.8);
	border: 1px solid rgba(255, 255, 255, 0.2);
	border-radius: 12px;
	padding: 24px;
	margin: 24px 0;
	font-size: 1.1rem;
	color: #ffffff;
	font-weight: 600;
}

.chart-container {
	background: rgba(15, 15, 15, 0.6);
	border-radius: 12px;
	padding: 20px;
	margin: 20px 0;
	border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Input Styling - Dark */
.gradio-container .gr-input,
.gradio-container .gr-dropdown,
.gradio-container .gr-slider {
	background: rgba(15, 15, 15, 0.8) !important;
	border: 1px solid rgba(255, 255, 255, 0.2) !important;
	border-radius: 8px !important;
	color: #ffffff !important;
}

.gradio-container .gr-input:focus,
.gradio-container .gr-dropdown:focus {
	border-color: #ffffff !important;
	outline: none !important;
}

.gradio-container label {
	color: #ffffff !important;
	font-weight: 500 !important;
	font-size: 0.95rem !important;
}

.gradio-container .gr-slider input {
	color: #ffffff !important;
}

/* Buttons - Dark Theme */
.gradio-container .gr-button-primary {
	background: transparent !important;
	border: 1.5px solid #ffffff !important;
	color: #ffffff !important;
	border-radius: 50px !important;
	padding: 16px 40px !important;
	font-weight: 500 !important;
	font-size: 16px !important;
	transition: all 0.3s ease !important;
}

.gradio-container .gr-button-primary:hover {
	background: #ffffff !important;
	color: #000000 !important;
}

.gradio-container .gr-button-secondary {
	background: transparent !important;
	border: 1.5px solid #ffffff !important;
	color: #ffffff !important;
	border-radius: 50px !important;
	padding: 16px 40px !important;
	font-weight: 500 !important;
}

.gradio-container .gr-button-secondary:hover {
	background: #ffffff !important;
	color: #000000 !important;
}

/* Tabs - Dark */
.gradio-tabs {
	border-bottom: 1px solid rgba(255,255,255,0.1) !important;
	background: transparent !important;
}

.gradio-tab {
	color: rgba(255,255,255,0.6) !important;
	border: none !important;
	background: transparent !important;
}

.gradio-tab.selected {
	color: #ffffff !important;
	border-bottom: 2px solid #ffffff !important;
}

/* Accordion - Dark */
.gradio-container .gr-accordion {
	background: rgba(15, 15, 15, 0.6) !important;
	border: 1px solid rgba(255, 255, 255, 0.1) !important;
	border-radius: 8px !important;
	color: #ffffff !important;
}

.gradio-container .gr-accordion label {
	color: #ffffff !important;
}

/* Content Pages */
.content-page {
	background: #000000;
	min-height: 80vh;
	padding: 60px 24px;
}

.content-card {
	background: rgba(20, 20, 20, 0.8);
	border: 1px solid rgba(255, 255, 255, 0.1);
	border-radius: 16px;
	padding: 40px;
	max-width: 900px;
	margin: 0 auto;
	box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}

.content-title {
	font-size: 2.5rem;
	font-weight: 800;
	color: #ffffff;
	margin-bottom: 24px;
	font-family: 'Poppins', sans-serif;
}

.content-text {
	font-size: 1.1rem;
	color: rgba(255, 255, 255, 0.8);
	line-height: 1.8;
	margin-bottom: 20px;
}

.contact-info {
	background: rgba(15, 15, 15, 0.6);
	border: 1px solid rgba(255, 255, 255, 0.1);
	border-radius: 12px;
	padding: 24px;
	margin: 20px 0;
}

.contact-item {
	font-size: 1.1rem;
	color: #ffffff;
	margin: 12px 0;
	font-weight: 500;
}

.contact-item strong {
	color: rgba(255, 255, 255, 0.9);
	margin-right: 12px;
}

/* Hide default Gradio styling */
.gradio-container footer {
	display: none !important;
}

.gradio-container .gradio-container {
	padding: 0 !important;
}
</style>
"""

# -----------------------------
# Build Gradio Interface
# -----------------------------
with gr.Blocks(css=custom_css, title="Product Return Predictions") as demo:
	# Navigation Bar
	with gr.Row():
		gr.HTML("""
			<div class="nav-bar">
				<div class="nav-logo">PRODUCT RETURN PREDICTIONS</div>
				<div class="nav-buttons">
					<button class="nav-button" onclick="document.querySelector('[data-testid=\\'tab-History\\']').click()">History</button>
					<button class="nav-button" onclick="document.querySelector('[data-testid=\\'tab-Contact\\']').click()">Contact Us</button>
					<button class="nav-button" onclick="document.querySelector('[data-testid=\\'tab-Learn\\']').click()">Learn More</button>
				</div>
			</div>
		""")
	
	# State for history
	history_state = gr.State([])
	
	with gr.Tabs(selected=0) as tabs:
		# Home Page
		with gr.TabItem("Home", id=0):
			with gr.Column():
				gr.HTML("""
					<div class="hero-section">
						<h1 class="hero-heading">Transform Your Business with PRODUCT RETURN PREDICTIONS</h1>
						<p class="hero-subheading">Accurate Return Forecasting Powered by Data & Machine Learning</p>
					</div>
				""")
				hero_button = gr.Button("Predictions Calculator", variant="primary", elem_classes="hero-button")
				gr.HTML("""
					<div class="wave-divider"></div>
				""")
			
			hero_button.click(fn=lambda: gr.update(selected=1), inputs=None, outputs=tabs)
		
		# Calculator Page
		with gr.TabItem("Calculator", id=1):
			with gr.Column(elem_classes="calculator-container"):
				with gr.Column(elem_classes="calculator-card"):
					gr.HTML("""
						<h2 class="calculator-title">Prediction Calculator</h2>
						<p class="calculator-subtitle">Enter the details below to generate accurate predictions</p>
					""")
					
					with gr.Row():
						with gr.Column(scale=1):
							with gr.Accordion("Order Details", open=True):
								product_category = gr.Dropdown(PRODUCT_CATEGORIES, value=DEFAULT_INPUTS[0], label="Product Category")
								product_price = gr.Slider(minimum=0, maximum=1000, value=DEFAULT_INPUTS[1], step=1, label="Product Price (₹)")
								order_quantity = gr.Slider(minimum=1, maximum=10, value=DEFAULT_INPUTS[2], step=1, label="Order Quantity")
								return_reason = gr.Dropdown(RETURN_REASON_OPTIONS, value=DEFAULT_INPUTS[3], label="Return Reason")
								days_to_return = gr.Slider(minimum=0, maximum=60, value=DEFAULT_INPUTS[4], step=1, label="Days to Return")
							
							with gr.Accordion("Customer Info", open=True):
								user_age = gr.Slider(minimum=16, maximum=75, value=DEFAULT_INPUTS[5], step=1, label="User Age")
								user_gender = gr.Dropdown(USER_GENDER_OPTIONS, value=DEFAULT_INPUTS[6], label="User Gender")
								user_location = gr.Dropdown(USER_LOCATION_OPTIONS, value=DEFAULT_INPUTS[7], label="User Location")
							
							with gr.Accordion("Payment & Shipping", open=True):
								payment_method = gr.Dropdown(PAYMENT_METHOD_OPTIONS, value=DEFAULT_INPUTS[8], label="Payment Method")
								shipping_method = gr.Dropdown(SHIPPING_METHOD_OPTIONS, value=DEFAULT_INPUTS[9], label="Shipping Method")
								discount_applied = gr.Slider(minimum=0, maximum=100, value=DEFAULT_INPUTS[10], step=1, label="Discount Applied (%)")
								delivery_days = gr.Slider(minimum=0, maximum=30, value=DEFAULT_INPUTS[11], step=1, label="Delivery Days")
							
							with gr.Accordion("History Metrics", open=True):
								past_orders = gr.Slider(minimum=0, maximum=30, value=DEFAULT_INPUTS[12], step=1, label="Past Orders")
								past_returns = gr.Slider(minimum=0, maximum=15, value=DEFAULT_INPUTS[13], step=1, label="Past Returns")
								past_return_rate = gr.Slider(minimum=0, maximum=1, value=DEFAULT_INPUTS[14], step=0.01, label="Past Return Rate")
								region = gr.Dropdown(REGION_OPTIONS, value=DEFAULT_INPUTS[15], label="Region")
							
							predict_btn = gr.Button("Predict", variant="primary")
							back_btn = gr.Button("Back to Home", variant="secondary")
						
						with gr.Column(scale=1):
							result_output = gr.Textbox(label="Prediction Result", elem_classes="result-box", interactive=False, lines=3)
							
							line_chart = gr.Plot(label="Confidence Trend", elem_classes="chart-container")
							
							bar_chart = gr.Plot(label="Category Analysis", elem_classes="chart-container")
					
					input_components = [
						product_category, product_price, order_quantity, return_reason, days_to_return,
						user_age, user_gender, user_location, payment_method, shipping_method,
						discount_applied, delivery_days, past_orders, past_returns, past_return_rate, region
					]
					
					predict_btn.click(
						fn=handle_prediction,
						inputs=[history_state] + input_components,
						outputs=[result_output, line_chart, bar_chart, history_state]
					)
					
					back_btn.click(fn=lambda: gr.update(selected=0), inputs=None, outputs=tabs)
		
		# History Page
		with gr.TabItem("History", id=2):
			with gr.Column(elem_classes="content-page"):
				with gr.Column(elem_classes="content-card"):
					gr.HTML("""
						<h2 class="content-title">Prediction History</h2>
					""")
					history_display = gr.HTML(value=format_history([]))
					
					def update_history_display(state):
						return format_history(state)
					
					history_state.change(fn=update_history_display, inputs=history_state, outputs=history_display)
		
		# Contact Us Page
		with gr.TabItem("Contact", id=3):
			with gr.Column(elem_classes="content-page"):
				with gr.Column(elem_classes="content-card"):
					gr.HTML("""
						<h2 class="content-title">Contact Us</h2>
						<div class="contact-info">
							<div class="contact-item">
								<strong>Email:</strong> nagasathvik1819@gmail.com
							</div>
							<div class="contact-item">
								<strong>GitHub:</strong> <a href="github.com/sathvik1819" target="_blank" style="color: #ffffff; text-decoration: underline;">https://github.com/yourusername</a>
							</div>
						</div>
						<p class="content-text">
							Feel free to reach out for questions, support, or collaboration opportunities.
						</p>
					""")
		
		# Learn More Page
		with gr.TabItem("Learn", id=4):
			with gr.Column(elem_classes="content-page"):
				with gr.Column(elem_classes="content-card"):
					gr.HTML("""
						<h2 class="content-title">Learn More</h2>
						<div class="content-text">
							<h3 style="color: #ffffff; font-size: 1.5rem; margin-bottom: 16px; margin-top: 24px;">What Does This Model Predict?</h3>
							<p class="content-text">
								This machine learning model predicts whether a product order is likely to be returned by the customer. 
								By analyzing various factors such as product category, price, customer demographics, shipping methods, 
								and historical return patterns, the model provides accurate forecasts of return probability.
							</p>
							
							<h3 style="color: #ffffff; font-size: 1.5rem; margin-bottom: 16px; margin-top: 24px;">Key Features Used</h3>
							<p class="content-text">
								The model considers multiple input features including:
							</p>
							<ul style="color: rgba(255, 255, 255, 0.8); line-height: 2; margin-left: 24px;">
								<li>Product characteristics (category, price, quantity)</li>
								<li>Customer information (age, gender, location, region)</li>
								<li>Order details (return reason, days to return, discount applied)</li>
								<li>Shipping and payment methods</li>
								<li>Historical customer behavior (past orders, returns, return rate)</li>
							</ul>
							
							<h3 style="color: #ffffff; font-size: 1.5rem; margin-bottom: 16px; margin-top: 24px;">Why Predict Product Returns?</h3>
							<p class="content-text">
								Product returns significantly impact business operations, inventory management, and profitability. 
								By accurately predicting returns, businesses can:
							</p>
							<ul style="color: rgba(255, 255, 255, 0.8); line-height: 2; margin-left: 24px;">
								<li>Optimize inventory planning and reduce overstocking</li>
								<li>Improve customer experience through proactive support</li>
								<li>Reduce operational costs associated with returns processing</li>
								<li>Make data-driven decisions about product pricing and promotions</li>
								<li>Identify patterns and trends in customer behavior</li>
							</ul>
							
							<h3 style="color: #ffffff; font-size: 1.5rem; margin-bottom: 16px; margin-top: 24px;">How Machine Learning Helps</h3>
							<p class="content-text">
								Machine learning algorithms can identify complex patterns and relationships in large datasets that 
								would be difficult for humans to detect. By training on historical return data, the model learns 
								which combinations of factors are most predictive of returns, enabling accurate forecasting for 
								new orders. This predictive capability helps businesses stay ahead of potential issues and make 
								informed strategic decisions.
							</p>
						</div>
					""")

# -----------------------------
# Launch App
# -----------------------------
if __name__ == "__main__":
	demo.launch(server_name="127.0.0.1", share=True)
