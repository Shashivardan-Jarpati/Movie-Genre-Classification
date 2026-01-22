from data_loader import load_data
from model import build_model

def predict_genre(plot_text):
    X, y = load_data()
    vectorizer, model = build_model()

    # Train on full dataset
    X_vec = vectorizer.fit_transform(X)
    model.fit(X_vec, y)

    # Predict new plot
    plot_vec = vectorizer.transform([plot_text])
    prediction = model.predict(plot_vec)

    return prediction[0]

if __name__ == "__main__":
    sample_plot = "A brave hero saves the world from an alien invasion"
    print("Predicted Genre:", predict_genre(sample_plot))
