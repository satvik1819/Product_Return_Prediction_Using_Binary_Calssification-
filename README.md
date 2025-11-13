Predicting Product Returns in E-commerce

Chapter -1 
Introduction:-
E-commerce has become one of the fastest-growing industries, offering customers convenience, variety, and competitive pricing. However, one of the major challenges faced by online retailers is product returns. High return rates directly affect profitability due to additional logistics, restocking, quality checks, and customer dissatisfaction.
Problem Statement :-
E-commerce platforms face huge financial losses due to frequent product returns. Returns not only affect revenue but also increase logistics cost, inventory mismanagement, and customer dissatisfaction. Currently, most platforms can only process returns after they happen, instead of preventing them.
So by this Binary Claassification model we are going to predict the products that are going to get returned .

Data Collection  and Data Description:-
We collected the data from Kaggle that have columns that can help us to train  , it has 10000 ROWS and 22 COLUMNS. The columns that we have in the dataset are :

Order_ID :-Unique identifier for each order placed.
Product_ID :-Unique identifier for each product.
 User_ID :-Unique identifier for each customer.
 Order_Date :-Date when the order was placed.
 Return_Date :-Date when the product was returned (if applicable).
Product_Category  :-The category/type of product (e.g., Electronics, Clothing, Home Appliances).
Product_Price :-Price of the product purchased.
Order_Quantity :-Number of units ordered for the product.
Discount_Applied :-Discount percentage or amount applied to the product.
Return_Reason :-Reason provided for returning the product (e.g., defective, wrong size, damaged).
Return_Status (Target Variable) :-Indicates whether the product was returned or not (Yes/No).
Days_to_Return :-Number of days taken by the customer to return the product.
User_Age :-Age of the customer.
User_Gender :-Gender of the customer (Male, Female, Other).
User_Location :-Geographical location of the customer.
Payment_Method :-Method of payment used (e.g., UPI, Credit Card, COD).
Shipping_Method :-Delivery option chosen (e.g., Standard, Express, Same-day).
Discount Applied :- How much discount was applied to each product.
Delivery Days:- Number of Days taken to Deliver the Prodcut.
Past Orders :- Count of Customers Past Orders.
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


Chapter-2 : Methodology

Introduction 
To predict product returns in e-commerce, we follow a structured machine learning pipeline. The methodology adopted in this project ensures that the data is carefully prepared, analyzed, and modeled to achieve reliable and interpretable results. The process involves the following key steps :

1.Data Understanding and Exploration
•	In this step we used various methods to load the data and used any cleaning techniques to understand the data.
•	Load and inspect the dataset.
•	Understand the meaning of each feature (customer details, product details, transaction details).
•	Perform Exploratory Data Analysis (EDA) to study data distributions, trends, and relationships with the target variable (Return_Status).
2.Data Cleaning
•	In this step we used many functions to identify and remove duplicate values , null values , missing values etc.
•	Handle missing values (imputation for numerical and categorical data).
•	Remove duplicate records.
•	Standardize inconsistent text entries (e.g., payment methods, gender).
•	Convert data types (dates, numerical values).
3.Feature Engineering 
•	Extract useful features from date columns (Order Month, Day of Week, Days to Return).
•	Create new features to reduce Redundency. 
•	Encode categorical features using label encoding and one-hot encoding.
•	Scale numerical variables for models that are sensitive to feature ranges.
4.Model Building 
•	Define the target variable (Return_Status).
•	Split the dataset into training and testing sets.
•	Train baseline and advanced classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost).
•	Tune hyperparameters to improve model performance.
5. Model Evaluation
•	Compare models using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
•	Pay special attention to Recall/F1-Score since identifying returned products is more critical than just overall accuracy.






Exploratory Data Analysis (EDA) Report

Exploratory Data Analysis (EDA) is the initial and most crucial step in any data science or machine learning project. It involves systematically examining the dataset to understand its structure, quality, and the relationships between variables.
The primary goal of EDA is to summarize the main characteristics of the data through both statistical methods and visual techniques, which helps in identifying patterns, spotting anomalies, detecting outliers, and forming hypotheses for further analysis.
1.Data Overview
•	The dataset contains 10,000 rows and 22 columns.
•	Columns include order details, product details, user details, transaction details, and return information.
•	The target variable is Return_Status (Yes/No).

2.Data Quality Check
•	Missing Values: Checked using .isnull().sum().
Some columns contain missing values, visualized using a bar chart.
•	Duplicates: Verified using .duplicated().sum().
Duplicates were identified and removed.
•	Data Types: Ensured that Order_Date and Return_Date were converted to datetime.
•	Spelling & Consistency: Verified categorical columns (e.g., Payment_Method, Shipping_Method, Return_Reason).
3.Target Variable Analysis
•	Return_Status distribution: Products are split into 1 (Returned) and 0 (Not Returned).
•	Helps understand class balance for classification modelling.
•	With the help of barplot we found that  Returned vs Not-Returned is 33 to 66. that is Returnned% = 33.5%  and Not-Returned = 66.5%.
4.Return Status vs Payment Method
•	I made a histogram to understand  how the mode of payment affect the Return rate and git the fillowing insights:
•	When the payments are made using the wallet the return rate was high
•	And when the mode of the payment is debit card and paypal the retun rate is low
•	And cash on delivery also have slightly higher return rates.
5.Return Rate vs Product Category 
•	Made another histogram that helps us to undersant how produt category is affecting the return rate and got the following insights:
•	Higest return rate is for the Clothing products and Asccessories 
•	And item used in home like decorations , tools have slightly higher return rate.
•	Books have moderately low return rate.
•	And Grocesaries have least return rate.
6.Delivery Days vs Return Status
•	Made a Histogram to understand how number of delivery days are effecting the return rate in the customers and got the following insights:
•	Return behavior by delivery time:
•	Orders delivered in 6–10 days have the highest return rate (37.7%).
•	Orders delivered in 2–6 days have the lowest return rate (32.2%).
•	Orders delivered in 0–2 days also have a significant return rate (34.9%).
•	For deliveries in 10–20 days, there’s missing data, so no conclusions can be drawn.
•	A longer delivery time might slightly increase returns, but it’s not a strict trend.
7.Produc Price vs Return Rate
•	 (4.329, 66.964] → 0.351: About 35.1% of orders in this price range were returned.
•	 (66.964, 113.132] → 0.320: About 32.0% were returned.
•	 (113.132, 155.157] → 0.339: About 33.9% were returned.
•	 (155.157, 198.786] → 0.348: About 34.8% were returned.
•	 (198.786, 243.04] → 0.330: About 33.0% were returned.
•	 (243.04, 289.462] → 0.322: About 32.2% were returned.
•	 (289.462, 338.973] → 0.348: About 34.8% were returned.
•	  (338.973, 390.844] → 0.346: About 34.6% were returned.
•	 (390.844, 445.382] → 0.321: About 32.1% were returned.
•	 (445.382, 567.2] → 0.325: About 32.5% were returned.
•	Return rates are fairly stable across price bins:
•	The return rate hovers between 32% and 35%.
•	There's no drastic increase or decrease in return rate as price increases.
•	Mid-range products may have slightly higher returns:
•	Some mid-priced products (like the third and seventh bins) show slightly higher return rates (34–35%).
•	Price may not be a strong factor influencing returns:
•	The return percentage doesn’t vary significantly between cheaper and expensive products.
8.Order Quantity vs Return Rate
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
•	  18–23 yrs → 34.41% returns (high)
•	23–25 yrs → 32.29% returns (slightly lower)
•	25–30 yrs → 35.50% returns (highest so far)
•	  30–32 yrs → 32.63% returns (drops again)
•	32–34 yrs → 35.53% returns (very high)
•	  34–44 yrs → 33.12% returns (mid)
•	44–46 yrs → 31.04% returns (low)
•	46–54 yrs → 31.32% returns (low)
•	54–63 yrs → 34.83% returns (high again)
•	63–67 yrs → 32.28% returns (low–mid)
Insights:
•	Younger users (18–23, 25–30, 32–34) have the highest return rates (35%).
Suggests impulsive buying or dissatisfaction.
•	Middle-aged groups (44–54 yrs) have the lowest return rates (31%).
Suggests more careful purchasing decisions.
•	Older users (54–63 yrs) again show higher return rates (34%).
 Possibly due to fit/quality issues or digital shopping challenges.

10.Past Orders vs Return Status
•	0–2 past orders → 33.35%
•	2–3 past orders → 33.60%
•	3–4 past orders → 32.94%
•	4–5 past orders → 32.91%
•	5–6 past orders → 34.48%
•	6–7 past orders → 34.99% (highest)
•	7–14 past orders → 34.13%
•	New customers (0–2 orders) already return at 33%.
 Suggests they aren’t much more cautious than repeat buyers.
•	Stable phase (3–5 orders) shows the lowest return rate (32.9%).
These users are likely more experienced/loyal, knowing what to expect.
•	High-frequency buyers (5+ orders) show increasing return rates (35%).
Could be because they experiment more, buy in bulk, or are less worried about returns.
11.Discount Applied vs Retunr Status
•	0–6% discount → 23.6% (lowest returns, very healthy)
•	 6–10% discount → 40.3% (huge jump)
•	10–14% discount → 40.8% 
•	14–19% discount → 34.6% (slight drop)
•	19–24% discount → 37.0% (rises again)
•	24–41% discount → 44.6% (highest)
•	41–70% discount → 43.4%
•	Low discounts (<6%) = lowest returns (24%)
Customers paying close to full price are more intentional & satisfied.
•	Medium discounts (6–14%) = sharp spike (40%)
 Suggests many impulse purchases that later get returned.
•	Very high discounts (24%+) = extremely high returns (44%)
 Could mean:
•	Products on clearance/low-quality stock being pushed.
•	Customers “try their luck” because of cheap deals.
•	Non-linear pattern: The 14–19% discount range is a small dip (34.6%), but overall higher discounts clearly correlate with higher returns.
•	Strongest predictive feature affecting the return rates.
12.Box Plots with Key Numeric Featurs vs Target Variable such as:
•	Product price vs Return rate 
•	User Age vs Return rate
•	Discount Applied vs return rate
•	Order quantity vs return rate
•	Insights:
•	Discount Applied (very strong)
•	Order Quantity (moderate, but weaker than discount)
•	Product Price (weak)
•	User Age (very weak)
13.HeatMap
•	Most variables are independent (low correlation).
•	Past_Returns and Past_Return_Rate are corelated with 0.79
•	Customers who have more past returns also naturally have a higher past return rate.
•	This is a derived relationship (since return rate is calculated from past returns / past orders).
•	Most other variables are weakly correlated (near zero).

Key Insights Of the EDA
1.	Clothing and Accessories face the highest return rates, mainly due to sizing issues.
The majority of returns are concentrated in clothing-related items, where “Wrong Size” is the leading reason. This highlights the need for more accurate size charts and virtual try-on guides.
2.	Electronics returns are driven by defects or damages.
Unlike clothing, electronics are returned primarily for product quality issues. This suggests stricter quality checks and safer packaging are necessary.
3.	Cash on Delivery (COD) orders are riskier, showing higher return rates.
COD customers often place impulsive or trial orders, leading to more returns compared to prepaid methods like Debit Card or PayPal.
4.	Standard shipping correlates with more returns than express or same-day delivery.
Longer delivery times increase dissatisfaction, while faster delivery reduces the likelihood of return.
5.	Younger customers (18–30 years) return more frequently than older groups.
Younger users tend to experiment more and return products easily, while middle-aged customers are more consistent in their purchases.
6.	Customers with high past returns are highly likely to return again.
Past behaviour is a strong predictor of future actions. Repeat returners form a distinct group and should be flagged for closer monitoring.
7.	High discounts lead to significantly higher return rates.
Products with 25%+ discounts show the maximum returns, reflecting impulsive and bulk buying behaviour during sales.
8.	Order quantity does not strongly influence returns.
Whether users buy single or multiple items, return rates remain mostly stable, with only a slight dip at moderate order sizes.
9.	Most returns happen within a week of delivery.
Customers quickly decide whether to keep or return an item, with very few returns after 10 days.
10.	Top return reasons: Wrong Size, Defective Item, and Delayed Delivery.
These three issues dominate return behaviour, pointing toward improvements needed in sizing accuracy, quality control, and logistics.





Chapter 3 : Feature Engineering And Model Building

Multicollinearity Problem :-
When analyzing our dataset using a correlation heatmap, we identified a significant issue of multicollinearity. Specifically, the variable Past_Return_Rate showed an extremely strong correlation (approximately 0.79) with Past_Returns. This means both features are providing almost the same information to the model. Multicollinearity creates problems in predictive modeling, especially in methods like logistic regression, where independent variables are assumed to contribute unique information. If two or more variables overlap heavily, the model may overemphasize redundant information, leading to unstable coefficients and unreliable predictions. Essentially, the model struggles to distinguish which feature is truly driving the outcome, making interpretation difficult and prediction performance weak.
Additionally, this redundancy inflates the importance of correlated features while suppressing the role of other independent predictors. In our case, keeping both Past_Returns and Past_Return_Rate could bias the model toward historical return patterns while ignoring other important factors such as product category, customer demographics, or payment method. This imbalance results in overfitting, reduced generalizability, and poor detection of minority cases like returned products. Identifying this problem in the heatmap was crucial, as it highlighted that some features must be engineered, transformed, or removed to ensure the model learns meaningful patterns instead of noise.

Rectifying The Mulitcollinearity Problem Using Feature Engineering :-
To address the multicollinearity issue revealed in our heatmap, we applied feature engineering techniques to refine our dataset before training the logistic regression model. The most direct solution was to remove the Past_Return_Rate variable, since it was almost a linear transformation of Past_Returns. By dropping this redundant feature, we reduced information overlap and ensured that the model focuses on unique and independent predictors. This not only improved interpretability but also stabilized the model’s coefficients, making predictions more reliable.
Beyond simply dropping correlated columns, we also considered balancing the target variable. Our confusion matrix showed that the model heavily favored “Not Returned” cases while failing to detect “Returned” items. To rectify this imbalance, we introduced feature engineering steps like class weighting in logistic regression and explored resampling methods such as SMOTE. These techniques helped the model pay more attention to minority cases, ensuring that Return_Status predictions became more balanced.
We also applied one-hot encoding to categorical variables such as product category, payment method, and user location, transforming them into machine-readable features without introducing artificial correlations. Together, these feature engineering strategies improved both fairness and predictive accuracy, enabling the model to generalize better and capture meaningful return behaviors.

Model Building
Phase -1 – Building Model
When we started working on our model, the first approach we tried was logistic regression. Since our goal was to predict whether a product would be returned or not, logistic regression felt like a natural starting point because it is simple, interpretable, and widely used for binary classification problems.
Before building the model, we cleaned up the dataset. We dropped the Past_Return_Rate column because it was highly correlated with Past_Returns, and keeping both could confuse the model. Then we converted all the categorical features like product category, payment method, and customer details into numerical values using one-hot encoding, so the model could actually process them. Once the dataset was ready, we split it into training and testing sets to make sure we could evaluate the performance fairly.
After training the model, the accuracy came out to be around 67%. At first, this looked promising because it suggested the model was doing better than random guessing. However, when we dug deeper into the confusion matrix and classification report, we noticed a major issue. The model was doing well at predicting “Not Returned” cases but almost completely failed at catching “Returned” products. This imbalance became the first serious challenge in our modeling process.

Phase-2-Problem with the MOdel
Although our first model ran without errors and showed a decent accuracy score, it had some serious shortcomings once we looked closer. The biggest problem came from the imbalance in the dataset. Most of the products in our data were not returned, while only a smaller portion were actually returned. Because of this, the model learned to play it safe by predicting “Not Returned” for almost everything. This is why the accuracy looked fine, but the recall for the “Returned” class was basically zero.
This issue became clear in the confusion matrix: the model correctly classified nearly all non-returns but completely missed the returns. From a business perspective, this is a big problem, because knowing which items are likely to be returned is far more valuable than simply predicting successful deliveries.
Another issue is that logistic regression itself is a linear model, which means it may not capture more complex relationships between variables like customer demographics, product category, and payment method. Even after cleaning the data and dropping redundant columns, the model was too simple to reflect the real-world patterns behind product returns.

Phase-3 – Solving the Problem with model 
After the initial attempts with Logistic Regression, we noticed a serious issue: the model was predicting almost only one class. Specifically, it would predict that products were not returned in nearly every case, completely ignoring the minority class of returned products. This was clear from the confusion matrix, where all the “returned” cases were misclassified, giving a recall of 0 for that class.
This happened because Logistic Regression struggles with imbalanced datasets, where one class is much more frequent than the other. In our dataset, the number of products not returned was significantly higher than those returned, so the model “learned” that predicting the majority class most of the time minimized error — but it failed to actually identify the products that were returned.
To address this, we switched to Random Forest, a model that is better suited for complex, non-linear relationships and can handle imbalanced datasets more effectively, especially when combined with SMOTE (Synthetic Minority Over-sampling Technique). SMOTE helps by creating synthetic samples of the minority class, balancing the dataset and allowing the model to learn patterns for returned products.
We also made sure to:
1.	One-hot encode categorical variables to convert them into numeric features.
2.	Add slight Gaussian noise to make the model more robust.
3.	Split the dataset into training and testing sets while preserving class ratios.
4.	Train the Random Forest with balanced class weights to further improve predictions for the minority class.
The results were significantly better. Our final model produced the following metrics on the test set:
•	Overall accuracy: 97%
•	Class 0 (Not Returned) recall: 100%
•	Class 1 (Returned) recall: 90%
•	F1-scores: 0.97 (class 0) and 0.95 (class 1)
The confusion matrix also showed that most products were correctly classified for both classes, with only a few returned products misclassified. This demonstrated that the Random Forest, combined with SMOTE and proper preprocessing, effectively solved the problem of imbalanced predictions and made the model practically useful for predicting product returns.







 





