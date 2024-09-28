import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag, word_tokenize
import nltk
from scipy.interpolate import griddata  # For interpolation

# Download necessary nltk resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Function to load data
def load_data(facts_file, fakes_file):
    try:
        with open(facts_file, 'r', encoding='utf-8') as f:
            facts = f.readlines()
        with open(fakes_file, 'r', encoding='utf-8') as f:
            fakes = f.readlines()
        facts = [fact.strip().strip('"') for fact in facts]  # Remove newlines and extra quotes
        fakes = [fake.strip().strip('"') for fake in fakes]
        labels = ['fact'] * len(facts) + ['fake'] * len(fakes)
        data = pd.DataFrame({'description': facts + fakes, 'label': labels})
        return data
    except OSError as e:
        print(f"Error opening or reading files: {e}")
        return None


# Text preprocessing function
def preprocess_text(text, method):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)  # Tokenize the text
    pos_tags = pos_tag(tokens)  # Get POS tags

    # Apply different preprocessing methods based on the argument
    if method == 'lemmatization':
        tokens = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags]
    elif method == 'stemming':
        tokens = [stemmer.stem(word) for word in tokens]
    elif method == 'pos_tagging':
        # Keep only nouns, verbs, and adjectives
        tokens = [word for word, tag in pos_tags if tag.startswith(('N', 'V', 'J'))]

    return ' '.join(tokens)


# Convert POS tag from nltk to WordNet POS for lemmatization
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'  # Adjective
    elif tag.startswith('V'):
        return 'v'  # Verb
    elif tag.startswith('N'):
        return 'n'  # Noun
    elif tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'  # Default to noun


# Feature extraction using TfidfVectorizer
def apply_preprocessing_and_feature_extraction(data, method, ngram_range=(1, 1)):
    data['processed_description'] = data['description'].apply(lambda x: preprocess_text(x, method))
    stop_words = list(stopwords.words('english'))  # Define stopwords
    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=ngram_range)  # Create TF-IDF features
    X = vectorizer.fit_transform(data['processed_description'])
    y = np.array([1 if label == 'fact' else 0 for label in data['label']])
    return X, y


# Lasso-based feature selection
def lasso_feature_selection(X_train, y_train, alpha=0.001):
    lasso = Lasso(alpha=alpha, random_state=42).fit(X_train, y_train)
    important_features = np.where(lasso.coef_ != 0)[0]  # Select non-zero coefficient features
    return X_train[:, important_features], important_features


# Function to select the classifier model
def select_model(model_name):
    if model_name == 'naive_bayes':
        return MultinomialNB()
    elif model_name == 'logistic_regression':
        return LogisticRegression(max_iter=1000)
    elif model_name == 'svm':
        return SVC()


# Function to split data into training, validation, and test sets
def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


# Model evaluation function for training, validation, and test sets
def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    # Evaluation on the training set
    train_predictions = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_predictions)
    train_precision = precision_score(y_train, train_predictions)
    train_recall = recall_score(y_train, train_predictions)
    train_f1 = f1_score(y_train, train_predictions)

    # Evaluation on the validation set
    val_predictions = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_predictions)
    val_precision = precision_score(y_val, val_predictions)
    val_recall = recall_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions)

    # Evaluation on the test set
    test_predictions = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions)
    test_recall = recall_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)

    # Print evaluation results
    print(
        f"Training Set - Accuracy: {train_acc:.2f}, Precision: {train_precision:.2f}, Recall: {train_recall:.2f}, F1 Score: {train_f1:.2f}")
    print(
        f"Validation Set - Accuracy: {val_acc:.2f}, Precision: {val_precision:.2f}, Recall: {val_recall:.2f}, F1 Score: {val_f1:.2f}\n")

    return {
        'train': {'accuracy': train_acc, 'precision': train_precision, 'recall': train_recall, 'f1': train_f1},
        'validation': {'accuracy': val_acc, 'precision': val_precision, 'recall': val_recall, 'f1': val_f1},
        'test': {'accuracy': test_acc, 'precision': test_precision, 'recall': test_recall, 'f1': test_f1}
    }


# Step 1: Find the best preprocessing method using split_ratio=0.2 and unigram analysis, based on accuracy
def experiment_find_best_preprocessing(data, models, preprocessing_methods):
    # Store the best preprocessing method for each model
    best_performance_per_model = {model: {'method': None, 'accuracy': 0} for model in models}
    method_count = {method: 0 for method in preprocessing_methods}  # To count how many times each method is the best
    highest_overall_accuracy = 0  # Track the highest accuracy across all models

    for method in preprocessing_methods:
        print(f"\nEvaluating Preprocessing Method: [{method}]")
        for model_name in models:
            X, y = apply_preprocessing_and_feature_extraction(data, method, ngram_range=(1, 1))
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=0.2)

            # Lasso feature selection
            X_train_selected, selected_features = lasso_feature_selection(X_train, y_train)
            X_val_selected = X_val[:, selected_features]
            X_test_selected = X_test[:, selected_features]

            model = select_model(model_name)
            model.fit(X_train_selected, y_train)

            print(f"Model [{model_name}], with pre-processing method [{method}]")
            model_results = evaluate_model(model, X_train_selected, y_train, X_val_selected, y_val, X_test_selected, y_test)
            val_accuracy = model_results['validation']['accuracy']  # focus on accuracy

            # Update the best preprocessing method for the current model based on accuracy
            if val_accuracy > best_performance_per_model[model_name]['accuracy']:
                best_performance_per_model[model_name]['method'] = method
                best_performance_per_model[model_name]['accuracy'] = val_accuracy

    # Count the frequency of each preprocessing method being the best
    for performance in best_performance_per_model.values():
        method_count[performance['method']] += 1

    # Find the method that was selected most often
    overall_best_method = max(method_count, key=method_count.get)

    # Print the best preprocessing method for each model
    for model_name, performance in best_performance_per_model.items():
        print(f"Best preprocessing method for model [{model_name}] is [{performance['method']}], with accuracy: {performance['accuracy'] * 100:.2f}% on the validation set")

    print(f"\nOverall best preprocessing method across models is [{overall_best_method}], chosen by {method_count[overall_best_method]} models.")

    return overall_best_method

# Step 2: Tune split ratio and n-gram using the best preprocessing method (with visualization)
def experiment_tune_parameters_with_visualization(data, best_method, models, split_ratios, ngram_ranges):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

    # Color maps for each model
    colormaps = ['viridis', 'plasma', 'inferno']  # Three distinct color maps
    best_hyperparams = {}  # To store the best hyperparameters for each model

    for i, model_name in enumerate(models):
        X_vals, Y_vals, Z_vals = [], [], []  # X for split ratios, Y for n-grams, Z for precision scores
        best_precision = 0
        best_params = {}

        for split_ratio in split_ratios:
            for ngram_range in ngram_ranges:
                X, y = apply_preprocessing_and_feature_extraction(data, best_method, ngram_range=ngram_range)
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=split_ratio)

                # Lasso feature selection
                X_train_selected, selected_features = lasso_feature_selection(X_train, y_train)
                X_val_selected = X_val[:, selected_features]
                X_test_selected = X_test[:, selected_features]

                model = select_model(model_name)
                model.fit(X_train_selected, y_train)

                precision = precision_score(y_val, model.predict(X_val_selected))

                # Store the values for 3D plotting
                X_vals.append(split_ratio)
                Y_vals.append(ngram_range[1])  # Use the upper value of the n-gram range
                Z_vals.append(precision)

                # Track the best precision and corresponding hyperparameters
                if precision > best_precision:
                    best_precision = precision
                    best_params = {'split_ratio': split_ratio, 'ngram_range': ngram_range}

        # Append model results for interpolation
        all_X_vals, all_Y_vals, all_Z_vals = X_vals, Y_vals, Z_vals

        # Perform grid interpolation to create a smooth surface
        grid_x, grid_y = np.mgrid[min(split_ratios):max(split_ratios):100j,
                                  min([n[1] for n in ngram_ranges]):max([n[1] for n in ngram_ranges]):100j]
        grid_z = griddata((X_vals, Y_vals), Z_vals, (grid_x, grid_y), method='cubic')

        # Plot the surface with different color maps for each model
        ax.plot_surface(grid_x, grid_y, grid_z, cmap=colormaps[i], alpha=0.6, label=f'{model_name}')

        # Store the best hyperparameters for the current model
        best_hyperparams[model_name] = best_params

    ax.set_xlabel('Split Ratio')
    ax.set_ylabel('N-gram')
    ax.set_zlabel('Precision')

    # Move the legend to the upper left and reduce the font size
    ax.legend(loc='upper left', fontsize='small')

    # Save the plot as a PNG file
    plt.savefig('hyper_parameter_tuning_visualization.png')
    plt.show()

    # Print the best hyperparameters for each model
    print("\nBest hyperparameters for each model:")
    for model_name, params in best_hyperparams.items():
        print(f"Model: {model_name}, Best Split Ratio: {params['split_ratio']}, Best N-gram Range: {params['ngram_range']}")


# Function to evaluate models with the best hyperparameters
def evaluate_best_model(data, best_hyperparams, models, best_split_ratio, best_ngram_range):
    print(f"\nEvaluating models with best parameters: Split Ratio: {best_split_ratio}, N-gram Range: {best_ngram_range}")

    best_model_name = None
    best_weighted_score = 0
    best_results = None

    for model_name in models:
        # Use the best preprocessing method (found earlier)
        X, y = apply_preprocessing_and_feature_extraction(data, best_hyperparams[model_name]['method'],
                                                          ngram_range=best_ngram_range)

        # Split data using the best split ratio
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=best_split_ratio)

        # Lasso feature selection
        X_train_selected, selected_features = lasso_feature_selection(X_train, y_train)
        X_val_selected = X_val[:, selected_features]
        X_test_selected = X_test[:, selected_features]

        # Train the model
        model = select_model(model_name)
        model.fit(X_train_selected, y_train)

        # Evaluate the model
        results = evaluate_model(model, X_train_selected, y_train, X_val_selected, y_val, X_test_selected, y_test)

        # Compute weighted score for validation set
        validation_weighted_score = (
            (results['validation']['accuracy'] + results['validation']['precision'] + results['validation']['recall']) / 3
        )

        # Print model performance
        print(
            f"Model: {model_name}, validation weighted score over Accuracy, Precision and Recall: {validation_weighted_score * 100:.2f}%\n"
        )

        # Track the best model based on the weighted score
        if validation_weighted_score > best_weighted_score:
            best_weighted_score = validation_weighted_score
            best_model_name = model_name
            best_results = results

    # Print the best model's performance
    print(f"\nBest Model: {best_model_name}")
    print(f"Accuracy: {best_results['test']['accuracy'] * 100:.2f}%")
    print(f"Precision: {best_results['test']['precision'] * 100:.2f}%")
    print(f"Recall: {best_results['test']['recall'] * 100:.2f}%")
    print(f"F1 Score: {best_results['test']['f1'] * 100:.2f}%\n")

# Main function for logic control
def main():
    # Define models and preprocessing methods
    models = ['naive_bayes', 'logistic_regression', 'svm']
    preprocessing_methods = ['lemmatization', 'stemming', 'pos_tagging']

    # Define hyperparameter ranges
    split_ratios = [0.1, 0.15, 0.2, 0.25, 0.3]  # Different train/test splits
    ngram_ranges = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]  # Different n-gram ranges

    facts_file = r'facts.txt'
    fakes_file = r'fakes.txt'

    # Load data
    data = load_data(facts_file, fakes_file)
    if data is None:
        print("Fact file or fake file does not exist.")
        return

    # Step 1: Find the best preprocessing method with split_ratio=0.2 and unigram analysis
    best_method = experiment_find_best_preprocessing(data, models, preprocessing_methods)

    # Step 2: Tune parameters and generate a 3D plot using the best preprocessing method
    experiment_tune_parameters_with_visualization(data, best_method, models, split_ratios, ngram_ranges)

    # The best hyperparameters from tuning (the result is calculated from Step 2)
    best_split_ratio = 0.15
    best_ngram_range = (1, 2)

    # Step 3: Evaluate all models using the best hyperparameters
    evaluate_best_model(data, best_hyperparams={'naive_bayes': {'method': best_method},
                                                'logistic_regression': {'method': best_method},
                                                'svm': {'method': best_method}},
                        models=models,
                        best_split_ratio=best_split_ratio,
                        best_ngram_range=best_ngram_range)

# Run the main function
if __name__ == '__main__':
    main()
