TABLE OF CONTENTS
Chapter 1 — Introduction
1.1 Overview of E-Commerce Returns
1.2 Problem Statement
1.3 Data Collection & Description
1.4 Pre-Processing Summary
Chapter 2 — Exploratory Data Analysis
2.1 Data Overview
2.2 Data Quality Check
2.3 Target Variable Analysis
2.4 Key EDA Visualizations
2.5 Important Insights
Chapter 3 — Feature Engineering & Model Building
3.1 Feature Engineering Steps
3.2 Encoding & Scaling
3.3 Models Used
3.4 Model Evaluation (Accuracy & F1 Score)
3.5 Best Model Selection
Chapter 4 — Modular Coding & Deployment
4.1 Modular Coding Structure
4.2 Data Ingestion
4.3 Data Transformation
4.4 Model Trainer
4.5 Utilities, Logger & Exception Handling
4.6 Gradio Deployment Overview
Chapter 5 — Future Scope & Conclusion
5.1 Future Scope
5.2 Conclusion
 
Predicting Product Returns in E-commerce

Chapter -1 
Introduction:-
E-commerce has become one of the fastest-growing industries, offering customers convenience, variety, and competitive pricing. However, one of the major challenges faced by online retailers is product returns. High return rates directly affect profitability due to additional logistics, restocking, quality checks, and customer dissatisfaction.
Product returns in e-commerce not only create operational complexities but also impact overall business efficiency. Unlike physical stores, online shoppers make purchase decisions without directly experiencing the product, which increases the chances of receiving items that do not meet their expectations in terms of size, quality, colour, or functionality. As a result, return rates in online retail are significantly higher, especially in categories such as clothing, accessories, and electronics. Each returned item triggers a reverse logistics cycle that includes transportation, inspection, repackaging, and, in some cases, disposal, all of which add financial strain on the company. Additionally, frequent returns disrupt inventory planning by creating inconsistencies between actual stock and predicted demand. This makes it difficult for businesses to maintain optimal inventory levels and fulfil future orders efficiently. From a customer perspective, a poor product experience or repeated returns can reduce trust in the platform, leading to lower retention and decreased long-term loyalty. Therefore, understanding the factors that lead to product returns is crucial for e-commerce businesses. By analysing customer behaviour, product attributes, transaction details, and fulfilment processes, companies can develop predictive models that help identify high-risk orders and implement strategies to reduce return rates and improve customer satisfaction.
Overall, understanding and predicting product returns has become an essential priority for modern e-commerce platforms. As online shopping continues to expand, the volume and complexity of returns will only increase, making traditional reactive approaches insufficient and costly. By integrating data-driven insights and machine learning techniques, businesses can shift toward a more proactive strategy that not only reduces return rates but also enhances customer experience and operational efficiency. Predictive models allow companies to identify high-risk orders, optimize product information, and refine logistical processes before issues occur. This leads to better decision-making, more accurate demand forecasting, and fewer disruptions across the supply chain. Ultimately, the goal is not only to minimize financial losses but also to build stronger customer trust and loyalty by ensuring that shoppers receive products that meet their expectations.
Problem Statement :-
One of the biggest challenges faced by e-commerce platforms today is the growing number of product returns. Every time a customer sends an item back, the company must handle extra work organizing reverse shipping, inspecting the product, repackaging it, or sometimes even discarding it. All of this adds significant cost and slows down overall operations. These returns also create confusion in inventory, making it harder for businesses to know what is truly in stock and plan future orders accurately. Many returns happen because customers receive products that don’t match their expectations in size, quality, or appearance, but companies currently deal with these issues only after the return is made. This reactive approach leads to unnecessary losses and missed opportunities to prevent returns before they occur. Since some customers, products, and order patterns are more likely to result in returns, there is a real need for a smarter, data-driven solution. By predicting which orders are at higher risk of being returned, e-commerce platforms can take early action such as improving product information, verifying orders, or enhancing packaging quality ultimately saving costs and improving the overall shopping experience for customers.

Data Collection  and Data Description:-
We collected the data from Kaggle that have columns that can help us to train , it has 10000 ROWS and 21 COLUMNS. The columns that we have in the dataset are :

Order_ID :-Unique identifier for each order placed.
Product_ID :-Unique identifier for each product.
 User_ID :-Unique identifier for each customer.
 Order_Date :-Date when the order was placed.
 Return_Date :-Date when the product was returned .
Product_Category  :-The category or type of product such as  Electronics, Clothing, Home Appliances .
Product_Price :-Price of the product purchased.
Order_Quantity :-Number of units ordered for the product.
Return_Reason :-Reason provided for returning the product such as defective, wrong size, damaged .
Return_Status (Target Variable) :-Indicates whether the product was returned or not (Retured / Not Returned) .
Days_to_Return :-Number of days taken by the customer to return the product.
User_Age :-Age of the customer.
User_Gender :-Gender of the customer (Male, Female, Other).
User_Location :-Geographical location of the customer.
Payment_Method :-Method of payment used such as  UPI, Credit Card, COD .
Shipping_Method :-Delivery option chosen such as Standard, Express, Same-day .
Discount Applied :- How much discount was applied to each product.
Delivery Days:- Number of Days taken to Deliver the Prodcut.
Past Orders :- Count of Customers Past Orders.
Past_Returns :- Count of the total products that customer returned in the past .
Past Return Rates :- Customers past return rate in percentage.
Region:- Larger Region classification of the customer North,Sounth,East,West.

Pre-Processing of the Data
Basically the dataset have 22 columns and 10000 rows which will help us to understand and build a working ML model that is going to predict if the product will be returned or not before the shipping . 
In the pre-processing we followed many steps to understand and also to clean the data the data by using many functions such as :

1.	Shape – We used shape to understand the dynamics of the dataset that is by using this attribute we got to know that our datset have 22 columns and 10000 rows.
2.	Describe-By using describe function we got mean , count , std , min ,max and many more outcome that can make us understand the numerical data from the dataset and by using same on categorical data we got frequencies of the every unique value of the columns.
3.	Isnull- By using the function isnull() we got the null and issing values of the columns such as Return_Date , Return_Reason , Days_to_Return having  close to 7000 missing values each. 
4.	Duplicated-By using the function duplicated we found out that we have 0 duplicated values in the dataset.
5.	Unique Values-We also found out frequency of unique values from every column for example the column User_Gender have two unique values Male and Female and the frequencies are Female-5052 and Male-4948.

Chapter-2 : Exploratory Data Analysis 

Exploratory Data Analysis (EDA) Report
Exploratory Data Analysis (EDA) is the initial and most crucial step in any data science or machine learning project. It involves systematically examining the dataset to understand its structure, quality, and the relationships between variables.
The primary goal of EDA is to summarize the main characteristics of the data through both statistical methods and visual techniques, which helps in identifying patterns, spotting anomalies, detecting outliers, and forming hypotheses for further analysis.


1.Data Overview
•	The dataset contains 10,000 rows and 22 columns.
•	Columns include order details, product details, user details, transaction details, and return information.
•	The target variable is Return_Status (Returned/Not Returned) .

2.Data Quality Check
•	Missing Values :Checked using .isnull().sum().
Some columns contain missing values, visualized using a bar chart.
•	Duplicates: Verified  using .duplicated().sum().
Duplicates were identified and removed.
•	Data Types: Ensured that Order_Date and Return_Date were converted to datetime.
•	Spelling & Consistency: Verified categorical columns (e.g., Payment_Method, Shipping_Method, Return_Reason).
3.Target Variable Analysis
•	Return_Status distribution: Products are split into 1 (Returned) and 0 (Not Returned).
•	Helps understand class balance for classification modelling.
•	With the help of barplot we found that  Returned vs Not-Returned is 33 to 66. that is Returnned percentage  = 33.5%  and Not-Returned percentage = 66.5%.
 

4.Return Status vs Payment Method
•	I made a histogram to understand  how the mode of payment affect the Return rate and git the fillowing insights:
•	When the payments are made using the wallet the return rate was high
•	And when the mode of the payment is debit card and paypal the return rate is low
•	And cash on delivery also have slightly higher return rates.
5.Return Rate vs Product Category 
•	Made another histogram that helps us to understand how product category is affecting the return rate and got the following insights:
•	Highest return rate is for the Clothing products and Accessories 
•	And item used in home like decorations , tools have slightly higher return rate.
•	Books have moderately low return rate.
•	And Groceries have least return rate.
6.Delivery Days vs Return Status
•	Made a Histogram to understand how number of delivery days are effecting the return rate in the customers and got the following insights:
•	Return behaviour by delivery time:
Delivery Days	Returns Rate
0-2	34.9%
2-6	32.2%
6-10	37.7%
10-20	NaN

7.Produc Price vs Return Rate
•	Made a Histogram to understand the relation between Product Price and Return Rate.
Product Price ($)	Return Rate
4-113	33.5%
113-199	34.3%
199-289	32.6%
289-391	34.7%
319-567	32.3%
•	Returns are moderate, showing that low-priced items are returned occasionally but not excessively.
•	slightly higher returns, likely due to high customer demand and expectation mismatches.
•	High-priced items have fewer returns, as buyers are more careful before purchasing.

8.Order Quantity vs Return Rate
Order Quantity	Product Rate
1-2	33.7%
2-3	33.8%
3-4	30.6%
4-6	33.6%
6-10	34.4%
•	Orders with 1 to 2 items have about 33.7% return rate. 
•	Orders  with 2 to 3 items have about 33.8% return rate. 
•	Orders with 3 to 4 items have about 30.6% return rate. 
•	Orders with 4 to 6 items have about 33.6% return rate. 
•	Orders with 6 ot 10 items have about 34.4% return rate. 
Insights:
•	Return rates are relatively consistent: The return percentage stays around 30–34% across all order quantities.
•	Slight dip at 3–4 items:Orders with 3–4 items show a slightly lower return rate (~30.6%) compared to other bins.
•	Higher quantities don’t always lead to more returns: Larger orders (6–10 items) don’t drastically increase returns; their return rate is similar to smaller orders.
9.User Age vs Return Rate
User Age	Return Rate
18-25	33.35%
25-32	34.52%
32-44	34.05%
44-54	31.18%
54-67	32.22%

Insights:
•	18–25 yrs: Moderate return rate due to exploratory and impulsive shopping.
•	25–32 yrs: Highest returns,maybe because of  by frequent fashion and discretionary purchases.
•	32–44 yrs: Returns remain high, indicating consistent high-volume online shopping behavior.
•	44–54 yrs: Lowest return rates, reflecting more intentional and accurate purchases.
•	54–67 yrs: Return rates rise again, likely due to sizing/mismatch or expectation gaps.

10.Past Orders vs Return Status
Past Orders	Return Rate
0-2	33.35%
2-3	33.60%
3-4	32.94%
4-5	32.91%
5-6	34.48%
6-7	34.99%
7-14	34.13

Insights :
•	0–2 past orders: Moderate returns, typical for new customers learning sizing/fit.
•	2–3 past orders: Slight increase, indicating early repeat buyers still adjust preferences.
•	3–4 past orders: Return rate dips as customers become more familiar with the platform.
•	4–5 past orders: Lowest rate, showing highly adapted and confident buyers.
•	5–6 past orders: Return rate rises again due to higher purchase volume or experimentation.
•	6–7 past orders: Highest return rate—frequent shoppers tend to try more items and return more.
•	7–14 past orders: Returns remain high, consistent with heavy and exploratory shopping behaviour.

11.Discount Applied vs Return Status
Discount Applied	Return Status
0-6%	23.6%
6-10%	40.3%
10-14%	40.8%
14-19%	34.6%
19-24%	37.0%
24-41%	44.6%
41-70%	43.4%
Insights:
•	0–6% discount has  lowest return rate  customers buying with low discounts are more certain about purchases.
•	6–10% discount  Returns jump sharply, indicating impulse buying driven by small discounts.
•	10–14% discount  Return rate stays high, showing discount-driven experimentation.
•	14–19% discount: Slight improvement  more balanced buyers in this range.
•	19–24% discount: Returns rise again as deal-seeking behavior increases.
•	24–41% discount: Highest return rate heavy discounting attracts high-risk, trial-based purchases.
•	41–70% discount: Returns remain extremely high due to bargain-hunting and low purchase commitment.
12.HeatMap
•	Most variables are independent (low correlation).
•	Past_Returns and Past_Return_Rate are corelated with 0.79
•	Customers who have more past returns also naturally have a higher past return rate.
•	This is a derived relationship (since return rate is calculated from past returns / past orders).
•	Most other variables are weakly correlated (near zero).

Key Insights from the EDA

•	The overall return rate is 33.5%, meaning 1 out of 3 products gets returned.
•	Wallet payments and Cash-on-Delivery orders show higher return rates compared to Debit Card and PayPal.
•	Clothing and Accessories have the highest return rates, while Groceries have the lowest.
•	Delivery days between 6–10 days show the highest return rate which  is about 37.7%, indicating delayed deliveries increase returns.
•	Product price has no strong linear impact on returns  return rates remain around 32–35% across all price ranges.
•	Order quantity does not significantly affect return behavior return rates stay stable around 33%.
•	Age group 25–44 shows the highest return rates, especially 25–32 years.
•	Customers with 6–7 past orders show the highest return rate which is about 35%, indicating frequent buyers return more.
•	Discounts have a strong effect high-discount orders (24%–70%) have extremely high return rates around 43%.
•	Correlation analysis shows almost no multicollinearity except Past_Returns vs Past_Return_Rate (0.79).








CHAPTER 3: FEATURE ENGINEERING  & MODEL BUILDING

Feature Engineering 
Before building and training the model I performed a little bit of Feature Engineering to improve the model performance.in this project many Feature Engineering teachniques were used to ensure and enhance the performance of the model .
1.Handling Missing values :
Using df.isnull().sum() we found there were few missing vales in the dataset , so I handled them either by filling them or removing them based on the importance of the specific column , for example if the column is an independent column surving less importance in the training process we simply dropped the column , on the other hand if the column is dependent clumn and important then :
•	Categorical missing vales were filled with a placeholder such as Unknown.
•	And Numerical missing values were filled with median .

2.Removing Duplicates :
Removing Duplicates prevents the model from overfitting over the repeated pattrens .
And to ensure it I removed all the duplicate values .

3.Encoding Categorical Columns :
Encoding categorical columns is important step in the process of feture engineering I did because the models can not understand the categorical values , so for this i preformed label encoding and One Hot encoding on all the categorical variables such as Product_Category , Payment_Method Shipping_Method , User_Location , Return_Reason , User_Gender .


4.StandardSclar for Numerical Columns :
For models like logistic regression scaling was done using standardsclar.
but for other complex models such as decision tree , XGBoost , CatBoost .


MODEL BUILDING

In this project I used multiple machine learning algorithams were used to train and evaluate the model , and also pick the best model out of all the models used based on the accuracy , F1 score .

We trained total of 7 algorithms such as :
1.	Logistic Regression
2.	Decision Tree
3.	Random Forest
4.	XGBoost
5.	Gradient Boosting
6.	LightGBM
7.	catBoost
We choose all Tree  Based algorithms because the dataset have both numerical and categorical values that needs to be trained for the model , and tree based models can handle both numerical and categorical and also non-linear data , and also have in built regression techniques such as  bootstrap , conrolled depth , split constraints , feature randomness.





Models Used :-

1.Logistic Regression
Logistic Regression is a  linear classification model used for binary outcomes. It estimates the probability of a class using the sigmoid function.   
I used this in the project to establish a baseline and understand initial predictive capability.
Advantages :
•	Simple to implement and interpret.
•	Performs well on linearly separable data.
•	Works well with large, sparse datasets.
Disadvantages:
•	Not suitable for complex, non-linear data.
•	Sensitive to outliers.
•	Requires feature scaling for optimal performance.

2.Desicion Tree
A tree-based model that splits data into branches based on feature thresholds. Creates a hierarchical structure where leaf nodes represent decisions (return/not return) .
Used this in the project to analyze simple tree-based decision logic for return prediction.
Advantages :
•	Can learn on Non-linear learning
•	Handles mixed numeric + categorical data
Disadvantages :
•	Overfits easily
•	Unstable to small data changes
3.Random Forest
Random forest is an algorithm that consists of multiple trained decision trees , while splitting the dataset into multiple subsets and each decision tree is trained on it to over come the overfitting problem .
Advantages :
•	Handles nonlinear interactions
•	Gives feature importance
•	Reduces overfitting
Disadvantages :
•	Can be slow with large numbers of trees
•	Not ideal for very high-cardinality categories

4.Gradient Boosting :
A boosting algorithm that builds trees sequentially, where each tree learns from the errors of the previous one.
Advantages :
•	High accuracy
•	Strong with non-linear data
Disadvantages :
•	Can overfit without tuning
•	Slower to train

5.XGBoost :
 A highly optimized and regularized version of gradient boosting , where every tree also learns from the mistaes of the previous tree , while using L1 and L2 regularization which are inbuilt to over come the overfitting problem .
Advantages :
•	Very fast and scalable
•	Excellent accuracy on tabular data
•	Reduces overfitting
Disadvantages :
•	Requires careful tuning
•	Sensitive to improper encoding of categories

6.LightGBM :
A high-performance gradient boosting framework developed by Microsoft that works by growing trees leaf-wisse instead of level wise and also uses histogram based learning to increase speed .
Advantages :
•	Fastest boosting model
•	High accuracy
•	Works well with large datasets
Disadvantage :
•	Occasional overfitting due to leaf-wise growth
•	Requires tuning for optimal performance

7.catBoost :
A gradient boosting algorithm specially designed to handle categorical features efficiently , by using inbuilt Ordered Target encoding .
Advantages :
•	Best for datasets with many categorical columns
•	No manual encoding required
Disadvantages :
•	Slower compared to LightGBM
•	Large model size


Model Selection  :
The model selection part is where  I selected best model out of all 7 models that I trained , for this I took accuracy and F1 score as the two main and important merits because  :
Accuracy :
Accuracy tells us about the overall predictions that were correct helping us to understand measure the general perforce of all the models .
F1 Score :
Tells us about the mean value of the precision and recall , which is important in my case because my dataset is an imbalanced dataset , this helps us in finding if model is biased towards predictiong only on one class.

BEST MODEL :
 

Considering the values of  both accuracy and F1 score I considered “catBoost” as my best model .
Because it achived a highest accuracy of 70.15% best amoung all the models .
And also it have the highest F1 score which is 0.421 , which means it performed well on both the classes without any bias .
CHAPTER 4 MODEL DEPLOYMENT

Once iam done with the model building and selection the best model I planned to deploy the model , for deploying the model I followed two basic steps :
1.Modular coding 
2.Platform to Deploy the model .

1.Modular Coding :
Modular coding is the part where I break the entire project into multiple independent modules such as :
1.	data_ingestion.py
2.	data_transfoormation.py
3.	model_trainer.py
This process helps for clean and workflow environment , for this I have to create a new environment variable in the local system . and made a separate folder for the project where I have all the components of the project as sub-folders and files .

I have multiple sub-folders under the src folder which is under the main folder named PRODUCT_RETURNS_PREDICTIONS_FINAL
Src have multiple folders such as COMPONENTS where I have all the important files for the modular coding such as :

Data_ingestion.py
The main responsibility of this file is to load the dataset and create a folder named artifacts to save train , test  datasets .
So basically it loads the raw csv file and saves the data into artifacts folder
Splits the data into 80-20 ratio saves both train.csv and test.csv in the artifacts foler .
And the artifacts folder also contains the pkl file of the model that we trained for the predictions .
 

Data_transformation.py :
The main purpose of this file is to preprocess raw data and convert it into machine-learning-ready format.
Loads train.csv and test.csv defines numerical and categorical lists and performs the Encoding and StandardSclar 
Drops unnecessary and independent columns , and perform all the required preprocessing steps to train and testing data.

Model_trainer.py :
The key responsibility of this file is to train multiple models and select the best model out of all the moels trained .
Imports and initializes seven ML models:
•	Logistic Regression
•	Decision Tree
•	Random Forest
•	Gradient Boosting
•	XGBoost
•	LightGBM
•	CatBoost

Fits each model using training data 	predicts using testing data .
And computes evaluation matrix that consists of accuracy and F1 score .
Logs all model performance selects the best model and saves the best model in the artifacts folder .

And I also have other files in the src folder such as :
Logger.py :
To record every important step of the ML pipeline for tracking and debugging.
•	Logs ingestion events
•	Logs transformation events
•	Logs model training progress
•	Logs errors and warnings
•	Saves logs into a dedicated log file

Exception.py :
To capture and manage errors in a clean and readable format.
•	Custom exception class for debugging
•	Provides meaningful error messages
•	Prevents pipeline from crashing
•	Helps track root-cause of errors during ingestion, transformation, or training
Utils.py :
To store reusable helper functions used across multiple modules.
•	Saves Python objects (pickle)
•	Loads saved objects
•	Supports evaluation functions if needed
•	Centralizes common tasks to avoid repeated code

DEPLOYMENT  OF MODEL
To deploy the model we need to choose a platform , for this I choose Gradio , to deploy this I made a file named app.py .
The file app.py loads the trained model that is saved in the artifacts folder 
And then many function were created to :
•	Convert user inputs into a DataFrame
•	Validate inputs using 10+ business rules
•	Reorder columns to match training data
•	Handle custom exception 

And also used css to design the complete multi page webpage .
Basically the webpage have multiple sub pages to make it user friendly 

 

The Prediction Calculator accepts all model features including:
Product Category, Price, Quantity, Return Reason, Days to Return, User Age, Gender, Location, Payment Method, Shipping Method, Discount Applied, Delivery Days, Past Orders, Past Returns, Past Return Rate, Region.
The interface uses dropdowns, sliders, and numeric inputs for user-friendly interaction.
The app generates two dynamic charts after prediction:
•	Line Chart – Confidence trend across predictions
•	Bar Chart – Return probability by category
A prediction history feature was implemented using Gradio to store the history of the past predictions including user details and displayed in tabular form .
And also everything from launching the app to storing the  errors everything will be logged .

The final machine learning solution was deployed using Gradio, which enabled a fully interactive and visually enhanced web interface. The deployment pipeline loads the trained model, validates inputs, generates predictions with confidence scores, and displays interactive charts. Multiple UI pages such as Home, Calculator, History, Learn More, and Contact Us were implemented using Gradio Blocks along with a custom dark-blue theme. The application can be run locally or publicly via Gradio’s built-in hosting using share=True, making the system user-friendly, visually appealing, and easily accessible.

 



Future Scope of the Project
1. Integrate Real-Time Data : Include live customer behaviour, delivery updates, and transaction logs to improve prediction accuracy continuously.
2. Build a Continuous Learning System : Retrain the model automatically as new orders and return data come in (AutoML / MLOps).
3. Add Deep Learning Models : Use Neural Networks, LSTMs, or Transformers to analyze sequential and customer behavior data.
4. Develop Return Risk Score : Provide a “Return Risk Score” instead of binary prediction for better business decision-making.
5. Add Explainable AI (XAI) :Use SHAP or LIME to explain WHY a product is likely to be returned (helps business understand key reasons).
6. Product-Specific Recommendations
Suggest actions like:
•	Offer size guidance
•	Change courier partner
•	Improve product quality
•	Modify pricing/discount strategy
7. Integrate with Inventory & Logistics Systems :Reduce restocking delays, optimize return centers, and plan reverse logistics using prediction results.
8. Personalized Customer Strategies : Identify high-return customers and apply:
•	Strict return policies
•	Personalized recommendations
•	Better product transparency
9. Agentic AI System
Build an AI Agent that:
•	Monitors return trends automatically
•	Alerts management
•	Suggests interventions
•	Auto-updates model
•	Generates reports autonomously
10. Multi-Language User Interface
Make the prediction tool accessible to more users by supporting Indian regional languages.


CONCLUSION :
This project successfully developed a complete end-to-end machine learning system capable of predicting product returns in an e-commerce environment. Through extensive Exploratory Data Analysis, key behavioral insights were identified across customer demographics, product characteristics, delivery factors, and discount patterns. Multiple machine learning algorithms—including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, and CatBoost—were trained and evaluated using Accuracy and F1 Score. CatBoost emerged as the best-performing model due to its superior handling of categorical data and consistent predictive performance.
A modular architecture was implemented to ensure scalability, maintainability, and ease of deployment. The pipeline included dedicated modules for data ingestion, preprocessing, transformation, model training, and prediction. The final model was deployed using a customized Gradio-based user interface, offering real-time prediction, interactive visualizations, prediction history, and a modern user experience.
This solution can provide immense value to businesses by predicting potential returns before they occur, reducing operational costs, improving customer experience, and enhancing inventory planning. The system sets a strong foundation for future enhancements, including real-time automation, deep learning integration, explainable AI, API deployment, and intelligent agent-based monitoring.
