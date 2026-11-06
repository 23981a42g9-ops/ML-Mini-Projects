import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

def train_spam_classifier():
    # Dummy dataset - replace 'data.csv' with a real dataset for your final GitHub repo
    data = pd.DataFrame({
        'message': [
            "Win a ₹1,00,000 cash prize now!", 
            "Are you joining the placement prep meeting tomorrow?",
            "URGENT: Your account is blocked. Click here to verify.",
            "Can we review the React project code this weekend?",
            "Exclusive offer! Buy 1 get 1 free on all shoes."
        ],
        'label': ['spam', 'ham', 'spam', 'ham', 'spam']
    })
    
    X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.4, random_state=42)
    
    # Create a machine learning pipeline
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    print("Model Evaluation:\n")
    print(classification_report(y_test, predictions, zero_division=0))
    
    # Test a custom input
    test_email = ["Let's discuss the database schema for Quantum Vidya."]
    print(f"Prediction for '{test_email[0]}': {model.predict(test_email)[0].upper()}")

if __name__ == "__main__":
    train_spam_classifier()