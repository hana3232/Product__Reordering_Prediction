# Product__Reordering_Prediction
Built a deep learning model to predict product reordering behavior using the Instacart dataset. Performed data preprocessing, feature engineering, and trained a DNN classifier. Achieved F1-score > 0.65. Helps in personalized recommendations, inventory planning, and customer retention
# 🧠 Deep Learning Based Product Reordering Prediction

## 📌 Project Overview

This project aims to **predict whether a customer will reorder a product** based on their past purchase behavior using **Deep Learning** techniques. The model is trained using the **Instacart Market Basket Analysis dataset**, which provides comprehensive details on customer orders, product metadata, and historical interactions.

### 🛒 Domain
- E-Commerce / Retail / Customer Analytics

---

## 🎯 Problem Statement

Given a user’s purchase history, predict the **likelihood of reordering a product** in their next order.

---

## 💼 Business Use Cases

- 📦 **Personalized Product Recommendation**: Suggest products likely to be reordered.
- 📊 **Inventory Management**: Forecast future demand based on reorder patterns.
- 🧍 **Customer Retention**: Identify churn based on changes in reorder behavior.
- 📢 **Marketing Optimization**: Target users with personalized campaigns based on reorder probability.

---

## 🧩 Dataset Details

The dataset used is from the [Instacart Market Basket Analysis](https://www.instacart.com/datasets/grocery-shopping-2017) competition. It consists of multiple CSV files:

| File | Description |
|------|-------------|
| `orders.csv` | Order details per customer with time info |
| `order_products__prior.csv` | Products from previous orders |
| `order_products__train.csv` | Products from training orders (with reorder flags) |
| `products.csv` | Product metadata |
| `aisles.csv` | Aisle metadata |
| `departments.csv` | Department metadata |

### Important Columns

- `user_id`, `product_id`, `order_id`
- `order_dow`, `order_hour_of_day`, `days_since_prior_order`
- `reordered` (Target: 0 = Not reordered, 1 = Reordered)

---

## 🔧 Project Approach

### 1. 📊 Data Understanding & EDA
- Joined datasets to build a user-product interaction matrix
- Explored user behavior and reorder patterns

### 2. 🛠️ Feature Engineering
- **User-level features**: total orders, reorder ratio, avg days between orders
- **Product-level features**: total times reordered, reorder probability
- **User-Product features**: times reordered by user, days since last order

### 3. 🧹 Data Preprocessing
- Handled missing values
- Encoded categorical variables (e.g., department, aisle)
- Normalized numerical features for model compatibility

### 4. 🤖 Deep Learning Model Building
- Built a **Feedforward Neural Network (DNN)** classifier using Keras
- Used **Binary Crossentropy** as the loss function
- Implemented **EarlyStopping** to prevent overfitting

### 5. ✅ Model Evaluation
- Evaluated performance using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
  - **ROC-AUC Score**
  - **Confusion Matrix**

### 6. 🚀 Model Deployment Ready
- Final trained model saved using `joblib`
- Prediction scripts created for inference

---

## 📈 Results

- Achieved **F1 Score > 0.65** on validation data
- Balanced performance across metrics for reorder prediction

---

## 🗂️ Project Structure


---

## 🧪 Tech Stack

- Python
- Pandas, NumPy
- Keras, TensorFlow
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 📝 Project Timeline

| Phase | Timeline |
|-------|----------|
| Data Understanding & EDA | Day 1 - Day 3 |
| Feature Engineering | Day 4 - Day 5 |
| Model Building | Day 6 - Day 7 |
| Evaluation & Tuning | Day 8 - Day 9 |
| Documentation & Delivery | Day 10 |

---

## 🧩 Project Evaluation Metrics

- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1 Score
- ✅ ROC-AUC Score
- ✅ Confusion Matrix

---

## 📜 License

This project is for educational purposes only.

---

## 🙌 Acknowledgements

Dataset provided by **Instacart**.

---

