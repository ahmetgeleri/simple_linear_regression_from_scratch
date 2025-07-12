import pandas as pd
import random
import math

#Load data from Excel
df = pd.read_excel('Coffee_Sales.xlsx')

#Convert DataFrame to list for manual processing
data = df.values.tolist()

#Shuffle and split data into training and test sets
random.seed(42)
random.shuffle(data)
split_index = int(len(data) * 0.7)
training_data = data[:split_index]
test_data = data[split_index:]

x1_train = [row[0] for row in training_data]  
x2_train = [row[1] for row in training_data]  
x3_train = [row[2] for row in training_data]  
y_train  = [row[3] for row in training_data]  

def pearson_correlation(x, y):
    n = len(x)
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    denominator = math.sqrt(
        sum((x[i] - x_mean)**2 for i in range(n)) *
        sum((y[i] - y_mean)**2 for i in range(n))
    )
    return numerator / denominator if denominator != 0 else 0

# Calculate correlations
x1_corr = pearson_correlation(x1_train, y_train)
x2_corr = pearson_correlation(x2_train, y_train)
x3_corr = pearson_correlation(x3_train, y_train)

print(f"x1 (Unit Price, y): {x1_corr:.4f}")
print(f"x2 (Quantity, y): {x2_corr:.4f}")
print(f"x3 (Discount_Amount, y): {x3_corr:.4f}")

# Choose the best feature with highest absolute correlation
correlations = [abs(x1_corr), abs(x2_corr), abs(x3_corr)]
best_index = correlations.index(max(correlations))
independent_variables = ['Unit Price', 'Quantity', 'Discount_Amount']
best_x_train = [row[best_index] for row in training_data]

print(f"\nBest correlated variable: {independent_variables[best_index]}")

# Linear Regression Coefficients
n = len(best_x_train)
x_mean = sum(best_x_train) / n
y_mean = sum(y_train) / n

b = (sum((best_x_train[i] - x_mean) * (y_train[i] - y_mean) for i in range(n)) 
     / sum((best_x_train[i] - x_mean)**2 for i in range(n)))
a = y_mean - b * x_mean

print(f"\na (intercept): {a}")
print(f"b (slope): {b}\n")

print(f"Regression Equation: ? = {a:.4f} + {b:.4f} * x\n")

y_train_pred = [a + b * x for x in best_x_train]
sse_train = sum((y_train[i] - y_train_pred[i])**2 for i in range(n))

print(f"Training SSE: {sse_train:.4f}")
print(f"Training Predictions (?): {y_train_pred}")

best_x_test = [row[best_index] for row in test_data]
y_test = [row[3] for row in test_data]
y_test_pred = [a + b * x for x in best_x_test]
sse_test = sum((y_test[i] - y_test_pred[i])**2 for i in range(len(y_test)))

print(f"\nTest SSE: {sse_test:.4f}")
print(f"Test Predictions (?): {y_test_pred}")

