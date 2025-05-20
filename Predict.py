from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import joblib

# Contoh dataset Iris
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, X_train)

# Simpan model ke file
joblib.dump(model, 'used_car_price_model_mae.pkl')