import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge

data = pd.read_csv('powerconsumption.csv')
data = pd.DataFrame(data)
data['Datetime'] = pd.to_datetime(data['Datetime'])
data['hour'] = data['Datetime'].dt.hour

def classify_time(hour):
    if 5 <= hour < 12:
        return "Morning"
    if 12 <= hour < 4:
        return "Afternoon"
    if 4 <= hour < 10:
        return "Evening"
    else:
        return "Night"

data['hour'] = data['hour'].apply(classify_time)

encoder = OrdinalEncoder(categories=[['Night', 'Morning', 'Afternoon', 'Evening']])
data['encoded_hour'] = encoder.fit_transform(data[['hour']])

data = data.drop(columns=['Datetime', 'hour'])
data_corr_matrix = data.corr()

print(data.columns)
X = data.drop(columns=['WindSpeed', 'PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3'])
y = data[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']]

#sns.heatmap(data_corr_matrix, cmap='coolwarm', annot=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
linear = LinearRegression()
linear.fit(X_train_scaled, y_train)
y_pred = linear.predict(X_test_scaled)
r2_score = linear.score(X_test_scaled, y_test)

print(r2_score)
ridge = Ridge(alpha=1, max_iter=1000)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
r2_score_ridge = ridge.score(X_test_scaled, y_test)
print(r2_score_ridge)

plt.scatter(y_test.iloc[:, 0], y_pred[:, 0], color='red')
plt.plot([y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()], [y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()], color='blue')

plt.show()
