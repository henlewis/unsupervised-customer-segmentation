# Unsupervised Customer Segmentation

An unsupervised machine learning project focused on segmenting retail customers using demographic, behavioural, geographic, and transaction-based features.

The project uses clustering techniques to identify meaningful customer groups, then applies association rule mining to uncover product combinations and recommend targeted marketing strategies for each segment.

---

## Project Overview

The objective of this project was to perform customer segmentation for a retail store using two datasets:

- Customer demographic and spending information
- Customer basket transaction data

The final goal was not only to cluster customers, but also to interpret each segment and translate the findings into practical business recommendations.

The project includes:

- Exploratory data analysis
- Feature engineering
- Missing value imputation
- Outlier detection
- Dimensionality reduction using PCA
- Clustering model comparison
- K-Means customer segmentation
- UMAP and PCA visualisation
- Customer profiling
- Association rule mining
- Segment-specific marketing strategy recommendations

---

## Dataset

The project uses two datasets.

### `customer_info.csv`

This dataset contains customer-level information.

| Feature Type | Examples |
|---|---|
| Customer ID | `customer_id` |
| Demographics | birth date, gender, kids at home, teens at home |
| Location | latitude, longitude |
| Shopping behaviour | distinct stores visited, typical shopping hour |
| Loyalty | loyalty card information |
| Spending history | lifetime spend by product category |
| Promotion behaviour | percentage of products bought on promotion |
| Complaints | number of formal complaints |

The dataset contains:

```text
34,060 customers
26 original columns
```

### `customer_basket.csv`

This dataset contains transaction-level basket data.

| Column | Description |
|---|---|
| `invoice_id` | Transaction identifier |
| `customer_id` | Customer identifier |
| `list_of_goods` | Products purchased in the basket |

The dataset contains:

```text
100,000 transactions
3 columns
```

---

## Business Problem

Retail customers often behave very differently depending on their lifestyle, household structure, spending power, preferences, and shopping habits.

A single marketing strategy is unlikely to work well for all customers.

This project aims to answer:

> Can we identify meaningful customer segments and use those segments to design more targeted marketing strategies?

---

## Project Workflow

```text
Raw Customer Data
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Missing Value Imputation
   ↓
Scaling
   ↓
Outlier Detection with DBSCAN
   ↓
Dimensionality Reduction with PCA
   ↓
Clustering Model Comparison
   ↓
Final K-Means Segmentation
   ↓
Cluster Profiling
   ↓
Association Rule Mining
   ↓
Targeted Marketing Recommendations
```

---

## Exploratory Data Analysis

The exploratory analysis focused on understanding customer behaviour and identifying useful patterns before clustering.

Key areas analysed included:

- Customer age distribution
- Gender distribution
- Number of children and teens at home
- Customer spending distributions
- Product category spending
- Shopping hour patterns
- Promotion usage
- Number of complaints
- Store visit behaviour
- Geographic distribution
- Most and least purchased items
- Feature correlations

Some important observations were:

- Spending features were highly right-skewed.
- Some customers had extremely high lifetime spending.
- Grocery spend and total lifetime spend contained large outliers.
- Meat and fish spending were strongly correlated.
- Customers with children and teens showed different household patterns.
- Shopping activity peaked around morning and early afternoon hours.
- Geographic plotting revealed a distinct customer group near a Makro-style business area.
- Basket data showed high-frequency essential products such as oil, cooking oil, and babies food.

---

## Feature Engineering

Several new features were created from the raw data.

### Customer Age

The original birth date column was converted into customer age.

```text
customer_birthdate → age
```

### Household Features

Kids and teens at home were combined into a total children feature.

```text
kids_home + teens_home → total_children
```

A binary feature was also created for customers with no children.

```text
no_kids
```

### Shopping Time Features

The typical shopping hour was converted into time-of-day behaviour indicators.

```text
typical_hour → morning_shopper
typical_hour → afternoon_shopper
typical_hour → evening_shopper
```

### Customer Duration

The year of first transaction was converted into customer relationship length.

```text
year_first_transaction → customer_for
```

### Loyalty Card

The loyalty card number was converted into a binary variable.

```text
loyalty_card_number → loyalty_card
```

### Education Level

Degree information was extracted from customer names and converted into an ordinal variable.

```text
None = 0
BSc = 1
MSc = 2
PhD = 3
```

### Spending Percentages

For each product category, the percentage of total lifetime spend was calculated.

Example:

```text
spend_meat_percent
spend_fish_percent
spend_groceries_percent
spend_videogames_percent
```

This allowed the clustering model to capture not only how much a customer spends, but also what they spend proportionally more on.

---

## Data Cleaning and Preprocessing

The preprocessing pipeline included:

- Removing irrelevant or redundant columns
- Converting birth date into age
- Converting loyalty card information into a binary feature
- Handling missing values with KNN imputation
- Correcting negative promotion percentage values
- Ensuring promotion percentages were within valid limits
- Encoding categorical variables
- Scaling numeric features using RobustScaler
- Separating a geographically distinct Makro/business customer group
- Removing temporary or visualisation-only features before modelling

Robust scaling was used because many spending columns contained outliers and skewed distributions.

---

## Outlier Detection

DBSCAN was used to identify unusual customers before the main clustering process.

A grid search was performed over different DBSCAN parameters.

Example parameter search:

| `eps` | `min_samples` | Outliers |
|---:|---:|---:|
| 4.3 | 4 | 193 |
| 4.4 | 6 | 184 |
| 4.5 | 6 | 161 |
| 4.6 | 6 | 142 |

The selected DBSCAN setup removed **184 outliers** before PCA and clustering.

These outliers were later assigned back to the nearest cluster centroid so that every customer had a final segment label.

---

## Dimensionality Reduction

PCA was used before clustering because the dataset contained many correlated numeric features.

The final PCA setup used:

```text
12 principal components
```

These 12 components explained approximately:

```text
88.52% of total variance
```

This reduced dimensionality while preserving most of the useful information for clustering.

---

## Clustering Model Selection

Several unsupervised clustering algorithms were tested and compared.

Models tested included:

- K-Means
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Model
- Self-Organising Map
- Mean Shift

The models were evaluated using:

- Silhouette Score
- Calinski-Harabasz Score
- Davies-Bouldin Score
- Number of clusters
- Noise points

### Model Comparison

| Algorithm | Silhouette | Calinski-Harabasz | Davies-Bouldin | Clusters |
|---|---:|---:|---:|---:|
| Hierarchical | 0.177 | 4452.3 | 1.751 | 8 |
| K-Means | 0.212 | 5002.5 | 1.541 | 8 |
| SOM | 0.069 | 2219.0 | 2.479 | 8 |
| DBSCAN | 0.070 | 10.8 | 0.962 | 8 |
| GMM | 0.171 | 4205.8 | 1.779 | 8 |

K-Means was selected because it achieved the best overall balance of clustering quality, interpretability, and business usefulness.

---

## Final Segmentation Approach

The final segmentation used K-Means, supported by PCA and additional business logic.

The workflow was:

1. Separate geographically distinct Makro/business customers.
2. Remove outliers using DBSCAN.
3. Apply PCA for dimensionality reduction.
4. Run K-Means clustering.
5. Re-run K-Means on remaining less well-defined groups.
6. Add the Makro customers as a separate business segment.
7. Assign removed outliers back to their nearest cluster.
8. Profile and name each final segment.

The final result was:

```text
7 customer segments
34,060 customers assigned
```

---

## Final Customer Segments

| Cluster | Segment Name | Customers |
|---:|---|---:|
| 1 | Family Shoppers | 5,920 |
| 2 | Tech & Gaming Enthusiasts | 3,758 |
| 3 | Balanced Budget Shoppers | 5,397 |
| 4 | Healthy Shoppers | 4,491 |
| 5 | Big Spenders | 8,815 |
| 6 | Students | 5,582 |
| 7 | Business Owner Makro | 97 |

---

## Segment Profiles

### Cluster 1: Family Shoppers

Family Shoppers had higher values for:

- Kids at home
- Total children
- Fish spending percentage
- Alcohol drinks spending percentage

This group appears to represent household shoppers buying for families.

### Cluster 2: Tech & Gaming Enthusiasts

This segment had higher values for:

- Lifetime videogame spending
- Videogame spending percentage
- Electronics spending
- Electronics spending percentage

This group appears to be younger or more entertainment-focused customers with strong interest in technology and gaming products.

### Cluster 3: Balanced Budget Shoppers

This group showed fewer extreme deviations compared with the other clusters.

They represent more moderate customers with practical shopping habits and balanced spending across categories.

### Cluster 4: Healthy Shoppers

Healthy Shoppers had higher values for:

- Vegetable spending
- Vegetable spending percentage
- Hygiene spending percentage
- Non-alcohol drinks spending percentage

They also showed lower spending on:

- Meat
- Fish
- Alcohol-related categories

This group appears to be more health-conscious and fresh-produce-oriented.

### Cluster 5: Big Spenders

Big Spenders had higher values for:

- Lifetime grocery spend
- Total lifetime spend
- Grocery spending percentage
- Total distinct products purchased

This was the largest customer segment and represents high-value customers for the business.

### Cluster 6: Students

Students had lower values for:

- Age
- Vegetable spending
- Videogame spending
- Lifetime vegetable spend

They also had a slightly higher number of complaints.

This segment was supported by geographic analysis showing a student-like cluster near a university area.

### Cluster 7: Business Owner Makro

This was a small but distinctive segment.

Business Owner Makro customers had higher values for:

- Total distinct products purchased
- Grocery spend
- Meat spend
- Fish spend
- Alcohol drinks spend

They had lower values for:

- Children at home
- Technology-related spending

This group likely represents business or wholesale-style shoppers.

---

## Cluster Visualisation

The final clusters were visualised using:

- PCA projections
- UMAP 2D visualisation
- UMAP 3D visualisation
- Cluster size bar charts
- Silhouette plots
- Geographic map plots
- Cluster profile heatmaps

The UMAP visualisation showed mostly clear separation between several customer groups, with some overlap between more general shopping segments.

---

## Association Rule Mining

After clustering, association rule mining was applied separately within each segment.

This allowed the project to identify product combinations that were especially relevant to each customer type.

The method used:

- TransactionEncoder
- Apriori algorithm
- Association rules based on lift, confidence, and support
- Cluster-specific minimum support thresholds

The rules were calculated separately for each customer segment to uncover more targeted product pairings.

---

## Example Association Rule Insights

### Family Shoppers

Common product relationships included:

- Minecraft and Ratchet & Clank
- Pokémon Sword and babies food
- Ratchet & Clank 2 and babies food

Marketing interpretation:

Family-oriented customers may buy both children’s products and household essentials, making them suitable for family bundles and cross-promotions.

### Tech & Gaming Enthusiasts

Common product relationships included:

- Energy bar and energy drink
- Protein bar and pancakes
- TikTok streaming gadgets and energy drink

Marketing interpretation:

This group is suitable for gamer-focused bundles, energy snack promotions, and tech accessory campaigns.

### Balanced Budget Shoppers

Common product relationships included:

- Muffins and tea
- Candy bars and tea
- Fresh bread and muffins

Marketing interpretation:

This group responds well to practical everyday discounts, simple bundle offers, and low-cost meal or snack combinations.

### Healthy Shoppers

Common product relationships included:

- Frozen vegetables and mashed potato
- Cauliflower and carrots
- Cauliflower and asparagus

Marketing interpretation:

This group is suitable for healthy recipe packs, fresh produce promotions, and nutrition-focused campaigns.

### Big Spenders

This group showed stronger interest in premium or higher-value products.

Marketing interpretation:

This segment is suitable for premium bundles, wine and food pairing events, VIP campaigns, and loyalty rewards.

### Students

Common product relationships included budget-oriented and social shopping combinations.

Marketing interpretation:

This group is suitable for student survival kits, weekend bundles, campus promotions, and social media campaigns.

### Business Owner Makro

This group showed wholesale-style and food-service-related purchasing behaviour.

Marketing interpretation:

This segment is suitable for bulk discounts, wholesale meal kits, and business-focused product bundles.

---

## Targeted Marketing Strategies

The final customer segments were translated into practical marketing recommendations.

| Segment | Example Strategy |
|---|---|
| Family Shoppers | Family bundles with children’s games and baby products |
| Tech & Gaming Enthusiasts | Gamer fuel kits with energy drinks and tech accessories |
| Balanced Budget Shoppers | Everyday essentials discounts and combo coupons |
| Healthy Shoppers | Healthy recipe packs and fresh produce campaigns |
| Big Spenders | Premium bundles, VIP loyalty tiers, wine and food events |
| Students | Student survival kits and weekend combo deals |
| Business Owner Makro | Bulk discounts, wholesale meal kits, and menu planning offers |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- scikit-learn
- PCA
- K-Means
- DBSCAN
- Agglomerative Clustering
- Gaussian Mixture Models
- Self-Organising Maps
- UMAP
- SciPy
- mlxtend
- Apriori association rules

---

## Repository Structure

```text
unsupervised-customer-segmentation/
│
├── data/
│   ├── customer_basket.csv
│   └── customer_info.csv
│
├── main.ipynb
├── model_selection.py
├── my_functions.py
├── visualizations.py
├── customer_clusters.csv
├── Final_report.pdf
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/unsupervised-customer-segmentation.git
cd unsupervised-customer-segmentation
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

On macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook main.ipynb
```

---

## Output Files

The project produces a final cluster assignment file:

```text
customer_clusters.csv
```

This file contains:

| Column | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| `cluster` | Assigned customer segment |

---

## Skills Demonstrated

This project demonstrates experience with:

- Unsupervised machine learning
- Customer segmentation
- Clustering model selection
- Feature engineering
- Data cleaning
- Missing value imputation
- Outlier detection
- Dimensionality reduction
- PCA
- UMAP visualisation
- K-Means clustering
- Cluster profiling
- Association rule mining
- Market basket analysis
- Business insight generation
- Data visualisation
- Translating machine learning results into marketing strategy

---

## Key Findings

The project identified seven interpretable customer segments:

1. Family Shoppers
2. Tech & Gaming Enthusiasts
3. Balanced Budget Shoppers
4. Healthy Shoppers
5. Big Spenders
6. Students
7. Business Owner Makro

The results showed that combining clustering with association rule mining can reveal both customer-level behaviour and product-level purchasing patterns.

This makes it possible to design more personalised marketing campaigns, such as family bundles, gamer packs, student discounts, healthy recipe kits, premium loyalty offers, and bulk business promotions.

---

## Future Improvements

Potential improvements include:

- Testing additional clustering algorithms such as HDBSCAN
- Using more advanced customer lifetime value features
- Adding recency, frequency, and monetary analysis
- Building an interactive dashboard for business users
- Automating cluster naming using feature importance
- Testing clustering stability over multiple random seeds
- Adding product category-level association rules
- Deploying the segmentation as a reusable customer scoring pipeline

---

## Authors

- Henry Lewis
- Yan Sidoryk
- Abdul Rehman Khan

---

## Project Status

Completed as part of a Machine Learning course project.

The final solution segments 34,060 customers into seven interpretable groups and uses association rules to generate segment-specific marketing recommendations.

---

## License

This repository is intended for educational and portfolio purposes.
