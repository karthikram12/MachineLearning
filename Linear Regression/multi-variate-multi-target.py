import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

data = pd.read_csv('powerconsumption.csv')
data = pd.DataFrame(data)
data['Datetime'] = pd.to_datetime(data['Datetime'])
data['hour'] = data['Datetime'].dt.hour

def classify_time(hour):
    if 5 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 22:
        return "Evening"
    else:
        return "Night"

data['hour'] = data['hour'].apply(classify_time)

encoder = OrdinalEncoder(categories=[['Night', 'Morning', 'Afternoon', 'Evening']])
data['encoded_hour'] = encoder.fit_transform(data[['hour']])

data = data.drop(columns=['Datetime', 'hour'])
data_corr_matrix = data.corr()

X = data.drop(columns=['WindSpeed', 'PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3'])
y = data[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']]

sns.heatmap(data_corr_matrix, cmap='coolwarm', annot=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def get_score(model, X_train1, X_test1, y_train1, y_test1):
    model.fit(X_train1, y_train1)
    return model.score(X_test1, y_test1)

scores_l = []

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test, y_train, y_test = X_scaled[train_index], X_scaled[test_index], y.iloc[train_index], y.iloc[test_index]
    scores_l.append(get_score(LinearRegression(), X_train, X_test, y_train, y_test))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

linear = LinearRegression()
linear.fit(X_train, y_train)
y_pred = linear.predict(X_test)
r2_score = linear.score(X_test, y_test)

cr =cross_val_score(linear, X_test, y_test)

ridge = Ridge(alpha=1, max_iter=1000)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
r2_score_ridge = ridge.score(X_test, y_test)

plt.scatter(y_test.iloc[:, 0], y_pred[:, 0], color='red')
plt.plot([y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()], [y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()], color='blue')

plt.show()
