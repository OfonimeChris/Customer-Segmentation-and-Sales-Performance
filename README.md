# Customer Segmentation and Sales Analysis

## Table of Content
- [Project Overview](#project-overview)
- [Data Sources](#data-sources)


### Project Overview
This data analysis project aim to provide insights into the customer segmentation and sales Performance of retail stores over a number of years. By analyzing various sections of the customer segment and sales, we seek to identify and group consumers with different needs and desires and to develop marketing applications and solutions specific to each group. We also identify trends, make more data driven recommendations and gain a deeper understanding of customer behavior and company’s performance in terms of revenue generated.
The project aims to uncover trends and patterns by delving into customer behavior and product trends. Customer Segmentation plays a critical role in many different areas. This include improving customer service, developing effective sales and distribution strategies, launching targeted promotional sales campaigns and enhancing cross-selling practices. The main motivation is driven by the need to improve business performance, customer experience, and overall profitability of the company through customer segmentation.


### Data Sources
The primary data used for this analysis is the "sales data.csv" file containing detailed information about each sales made by the company. It’s a Walmart Retail Dataset, and the source is from Data world

### Tools
- Excel - data cleaning
- Python -data cleaning and anaysis
- Tableaux - creating Reports

  

### Data Cleaning/ Preparation
 in the initial data phase, we performed the following task:
 - Data download and inspection
 - Handling missing values,duplicate removed and correction of inconsistencies (e.g., outliers, incorrect data formats)
 - Data cleaning and formatting

### Exploratory Data Analysis
EDA involves exploring the sales data to answer key questions:
    ### Customer Segmentation
    -How are customers distributed across different segments (e.g., customer_segment, region, state, city)
    -How does customer age correlate with sales, order quantity, or discount usage?
    -Which product categories or subcategories are preferred by different customer segments?

     ###Sales Analysis
    -what is the overall sales trend
    -What are the top-performing regions, states, or cities in terms of sales and profit?
    -How do sales and profit vary by product category, sub-category, or container type?
    -Which customer segments are the most and least profitable?
    -What is the average order quantity, and does it differ significantly across regions or customer segments?
    -How do discounts impact sales and profit margins?

    ###Retention and Loyalty
    -How frequently do customers place orders, and what is the average time between purchases?
    -Are there any patterns in customer purchasing behavior over time?
    -Which customer segments have the highest lifetime value (e.g., cumulative sales/profit)?
    -What is the relationship between shipping cost, mode, and customer retention?
    -Are high-priority orders associated with better profitability or retention?

     ###Revenue Growth Optimization
    -Which product categories or subcategories offer the highest margins?
    -How do shipping costs affect profitability across different regions or segments?
    -Are there correlations between order date (seasonality) and sales or profit?
    -Which regions or customer segments respond most positively to discounts?
    -What are the trends in unit price and product margin, and how do they relate to profit?

### Data Analysis
##### Data Preprocessing: 
   -I Visualized correlations between features and key outcomes like customer behavior or sales performance. Also Identified trends and patterns across different customer 
    segments, regions, or time frames. This will give me an understanding of the data structure.
#### Association Rule: 
   -During segmentation, association rule aid in identifying products that are frequently bought together, which is useful for cross-selling strategies, product bundling, 
    and improving inventory management. So, for Walmart Grocery, it can aid in the optimization of store layouts and promotions by placing frequently bought items together 
    or offering discounts on complementary products.
##### Feature engineering: 
  -Feature scaling, which is a necessary step before performing clustering as KMeans is sensitive to feature scale. This ensures that all features have equal weight in the 
   clustering algorithm. 
   Creating new features to better capture data structure.
#### Clustering: 
  -I Used the KMeans algorithm to perform clustering. Two Number of clusters was chosen based on the Elbow method result, which measures the quality of the clustering.
   Cluster Interpretation and Visualization: Interpretation of the clusters by examining the average values of the features within each cluster. Will also visualized the 
   clusters using scatter plots.
#### Models: 
  -Application ‘RFM analyses for customer segmentation or use clustering algorithms (K-means) to group customers. Built classification models using random Forest Classifier 
   and Logistics regression model to predict customer churn.
  -I carried out a based model to accertain the best hyperparmeter to be used on three different algorithms to compare their accuracy and  employ the best algorithm to 
   predict sales trend, which were XGBoost, Random Forest regression and Linear regression. XGBoost came out as the 
   best algorithm for predicting  sales trend and a model for forcasting sales performance was developed for future use.
   
### Some codes languages and feautures i worked with
- Python

  from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV


# Preprocessing pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

models = {
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 5, 10]
        }
    },
    'XGBoost': {
        'model': XGBRegressor(random_state=42, use_label_encoder=False),
        'params': {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
            'regressor__max_depth': [3, 6, 10]
        }
    },
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {}
    }
}



# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_models = {}

for name, entry in models.items():
    print(f"Training model: {name}")
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', entry['model'])
    ])
    
    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        model,
        entry['params'],
        cv=3,  # 3-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_models[name] = (best_model, best_params)
    
    # Evaluate the model
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Best Parameters for {name}: {best_params}")
    print(f"{name} - Mean Squared Error: {mse:.4f}")
    print(f"{name} - R^2 Score: {r2:.4f}\n")

  ### Result
  Training model: Random Forest
  Best Parameters for Random Forest: {'regressor__max_depth': 10, 'regressor__min_samples_split': 5, 'regressor__n_estimators': 50}
  Random Forest - Mean Squared Error: 4709075.6124
  Random Forest - R^2 Score: 0.8017

  Training model: XGBoost
  Best Parameters for XGBoost: {'regressor__learning_rate': 0.2, 'regressor__max_depth': 3, 'regressor__n_estimators': 200}
  XGBoost - Mean Squared Error: 869792.9593
  XGBoost - R^2 Score: 0.9634

  Training model: Linear Regression
  Best Parameters for Linear Regression: {}
  Linear Regression - Mean Squared Error: 17394971.5213
  Linear Regression - R^2 Score: 0.2675


- Tableau

### Results and Findings
    The analysis/results are summarized as follows:
    When analyzing Churn,  i realized that the model achieved a perfect accuracy of 1.0000, indicating that it correctly classified all instances influence by 
    Day_since_last_order column,  Day_since_last_order was the most important feature suggests that customer inactivity is a strong predictor of churn. This aligns with 
    intuition, as customers who haven't made purchases in a long time are more likely to churn. other features like "profit," "shipping cost," and "customer age" likely 
    contribute to the model's overall performance as well.

-Recommendation: Company can segment customers based on their churn risk and tailor retention strategies accordingly. Also can offer personalized incentives, discounts, or loyalty programs to retain high-value customers. Customer Feedback can also be gotten to understand reasons for churning and identify areas for improvement.
-The company had a slow sales with 0-2000 over the years with a spike during holiday periods, I also discovered that the home office customer segment drove the most revenue and the central and east region also make the higest sales performance.
marketing stretegies should be promoted in these segment and regions


-Product Category A is the best categorgy in termsof generating sales and revenue
-Customer Segment with high life time value should be targeted for marketing efforts

### Recommendations
  Based on the analysis, I will recommend the following actions to be taken:
  1. Customers with segment based on their churn risk, retention strategies should be targeted to them accordingly.
  2. invest in marketing and promotions during peak sales seasons to maximize revenue.
  3. focus expanding and promoting product category that crings more revenue
  4. focus on expanding and promoting regions with mega sales and implement same stretegies to regions with low sales for better sales performance
  5. implement a customer segmentation strategy to target high life time customer effectively
  6. Since the home office customer segment drove the most revenue and the central and east region also made the higest sales performance. it is wise to recomment 
     marketing and retention  stretegies to be promoted in these segment and regions.
  7. Personalized incentives, discounts, or loyalty programs should be offered to retain high-value customers.

  8. Company should actively seek feedback from customers to understand their reasons for churning and identify areas for improvement.

  9. Products should be recommended to customers based on their purchase history and the preferences of similar customers.

  10. Walmart should adjust prices based on demand, competition, and other factors to maximize revenue.

  11. Company should strategically place products to maximize visibility and impulse purchases.

  12. Creation of a pleasant shopping environment that encourages customers to spend more time and money.

  13. Implementation of loyalty programs to reward repeat customers and encourage repeat purchases.

### Limitations

### References


​












