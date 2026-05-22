import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
n = 5000

# ── Synthetic e-commerce features ────────────────────────────────────────────
time_on_site    = np.random.exponential(5, n).clip(0.5, 60)      # minutes
pages_visited   = np.random.poisson(4, n).clip(1, 30).astype(float)
items_in_cart   = np.random.poisson(2, n).clip(1, 20).astype(float)
discount_pct    = np.random.choice([0, 5, 10, 15, 20, 30], n,
                                    p=[0.4, 0.15, 0.2, 0.1, 0.1, 0.05]).astype(float)
is_returning    = np.random.binomial(1, 0.35, n).astype(float)   # 1 = returning customer
device_mobile   = np.random.binomial(1, 0.55, n).astype(float)   # 1 = mobile
hour_of_day     = np.random.randint(0, 24, n).astype(float)
avg_item_price  = np.random.uniform(10, 500, n)                   # R$ per item

# ── Target: order value (R$) ─────────────────────────────────────────────────
order_value = (
    items_in_cart * avg_item_price * (1 - discount_pct / 100)
    + time_on_site * 3.5
    + pages_visited * 8
    + is_returning * 60
    - device_mobile * 20
    + hour_of_day * 1.2
    + np.random.normal(0, 40, n)
).clip(10)

feature_names = [
    'time_on_site', 'pages_visited', 'items_in_cart',
    'discount_pct', 'is_returning', 'device_mobile',
    'hour_of_day', 'avg_item_price',
]

X = np.column_stack([
    time_on_site, pages_visited, items_in_cart,
    discount_pct, is_returning, device_mobile,
    hour_of_day, avg_item_price,
])
y = order_value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}  |  R²: {r2:.4f}")

np.savez(
    'model_params.npz',
    weights=model.coef_,
    bias=np.array([model.intercept_]),
    feature_names=np.array(feature_names),
)
print("model_params.npz saved.")