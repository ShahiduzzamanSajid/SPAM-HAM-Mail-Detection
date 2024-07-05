# Real-Time SPAM HAM Mail Detection


## Project Overview
This project is a spam mail detection system built using machine learning and Streamlit. The application allows users to test if an email is spam, add training data dynamically, and evaluate the model's performance. The model uses the MultinomialNB algorithm from scikit-learn, and additional features are extracted from the email text to improve detection accuracy. 

## Features
1. **Email Classification :** Classify emails as "Spam" or "Ham" based on the content.
2. **Influential Words Highlighting :** Display the most impactful words contributing to the classification.
3. **Test an Email :** Users can input an email and the system will classify it as spam or not spam.
4. **Add Training Data :** Users can add new emails and label them as spam or not spam, which will be used to retrain the model.
5. **Evaluate Model :** Users can evaluate the model's performance on a test dataset.

## Libraries and Tools Used
- **Streamlit :** A framework for building interactive web applications.
- **Pandas :** A data manipulation and analysis library for Python.
- **NumPy :** A library for numerical operations in Python.
- **Scikit-Learn :** A machine learning library for Python. It provides tools for model building, feature extraction, and evaluation.
  - **Features Used :**
    - **TfidfVectorizer :** Converts text into a matrix of TF-IDF features for text classification.
    - **MultinomialNB :** A Naive Bayes classifier for multinomially distributed data, used to train the spam detection model.
  - **Model Training :** Multinomial Naive Bayes classifier is employed to build and train the spam detection model.
- **Imbalanced-Learn :** A library for handling imbalanced datasets.
  - **Techniques Used :** SMOTE is used to balance the class distribution in the dataset.
- **Pickle :** A module for serializing and deserializing Python objects. It is used for saving and loading the trained model and vectorizer.


## Contributing
Contributions to improve Real-Time Spam ham Detection are welcome. Please fork the repository, make your changes, and submit a pull request.
