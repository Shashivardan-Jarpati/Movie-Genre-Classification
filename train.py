from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_loader import load_data
from model import build_model

def train_model():
    # Load data
    X, y = load_data()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build model
    vectorizer, model = build_model()

    # Vectorize text
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model.fit(X_train_vec, y_train)

    # Test model
    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)

    print("✅ Model training completed")
    print("🎯 Accuracy:", accuracy)

if __name__ == "__main__":
    train_model()
