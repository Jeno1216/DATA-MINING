from flask import Flask, render_template, request, jsonify, redirect
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re
from tensorflow.keras.preprocessing.text import tokenizer_from_json



import json
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)



# landing page
@app.route('/')
def index():
    return render_template('index.html')

# landing page
@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/team')
def team():
    return render_template('team.html')



# load the saved model for KNN
knn_model = pickle.load(open('model.pkl', 'rb'))

@app.route('/knn_predict', methods=['GET', 'POST'])
def knn_predict():
    if request.method == 'POST':
        # get the form data from the request
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        # create a new sample as a numpy array
        new_sample = np.array([feature1, feature2]).reshape(1, -1)

        # make a prediction using the loaded model
        y_pred = knn_model.predict(new_sample)

        if y_pred[0] == 1:
            prediction_text = 'Junior'
        elif y_pred[0] == 2:
            prediction_text = 'Senior'
        elif y_pred[0] == 3:
            prediction_text = 'Project Manager'
        else:
            prediction_text = str(y_pred[0])

        # return the prediction result as JSON
        return jsonify({'prediction': prediction_text})

    # handle GET request
    return render_template('/knn_predict.html')



# load the saved model for gaussian naivebayes
gnb_model = pickle.load(open('indeed_gnb_model.pkl', 'rb'))

@app.route('/gaussianNB', methods=['GET', 'POST'])
def gaussianNB():
    if request.method == 'POST':
        # get the form data from the request
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        # create a new sample as a numpy array
        new_sample = np.array([feature1, feature2]).reshape(1, -1)

        # make a prediction using the loaded model
        y_pred = gnb_model.predict(new_sample)

        if y_pred[0] == 1:
            prediction_text = 'Junior'
        elif y_pred[0] == 2:
            prediction_text = 'Senior'
        elif y_pred[0] == 3:
            prediction_text = 'Project Manager'
        else:
            prediction_text = str(y_pred[0])

        # return the prediction result as JSON
        return jsonify({'prediction': prediction_text})

    # handle GET request
    return render_template('/gaussianNB.html')

bnb_model = pickle.load(open('indeed_bnb_model.pkl', 'rb'))
@app.route('/bernoulliNB', methods=['GET', 'POST'])
def bernoulliNB():
    if request.method == 'POST':
        # get the form data from the request
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        # create a new sample as a numpy array
        new_sample = np.array([feature1, feature2]).reshape(1, -1)

        # make a prediction using the loaded model
        y_pred = bnb_model.predict(new_sample)

        if y_pred[0] == 1:
            prediction_text = 'Junior'
        elif y_pred[0] == 2:
            prediction_text = 'Senior'
        elif y_pred[0] == 3:
            prediction_text = 'Project Manager'
        else:
            prediction_text = str(y_pred[0])

        # return the prediction result as JSON
        return jsonify({'prediction': prediction_text})

    # handle GET request
    return render_template('/bernoulliNB.html')

mnb_model = pickle.load(open('indeed_mnb_model.pkl', 'rb'))
@app.route('/multinomialNB', methods=['GET', 'POST'])
def multinomialNB():
    if request.method == 'POST':
        # get the form data from the request
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        # create a new sample as a numpy array
        new_sample = np.array([feature1, feature2]).reshape(1, -1)

        # make a prediction using the loaded model
        y_pred = mnb_model.predict(new_sample)

        if y_pred[0] == 1:
            prediction_text = 'Junior'
        elif y_pred[0] == 2:
            prediction_text = 'Senior'
        elif y_pred[0] == 3:
            prediction_text = 'Project Manager'
        else:
            prediction_text = str(y_pred[0])

        # return the prediction result as JSON
        return jsonify({'prediction': prediction_text})

    # handle GET request
    return render_template('/multinomialNB.html')




# KMEANS
# FOR CREATING CLUSTERS AND DISPLAY IMAGE 
# Load the data
data = pd.read_csv('filled_indeed.csv')

# Cluster Experience Required and Salary
X = data.iloc[:, [2, 3]].values
# Cluster Job Title and Salary
Y = data.iloc[:, [0, 3]].values

# Load the KMeans model using pickle
with open('kmeans_model.pickle', 'rb') as f:
    loaded_model = pickle.load(f)

# Predict the clusters using the loaded model
# Predict the clusters using the loaded model
y_kmeans_loaded_X = loaded_model.predict(X)
y_kmeans_loaded_Y = loaded_model.predict(Y)

# Generate the first cluster plot and save it to a file
plt.scatter(X[y_kmeans_loaded_X == 0, 0], X[y_kmeans_loaded_X == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans_loaded_X == 1, 0], X[y_kmeans_loaded_X == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans_loaded_X == 2, 0], X[y_kmeans_loaded_X == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], s=100, c='black', marker='x', label='Centroids')
plt.title('Clusters of Data (Experience Required vs Salary)')
plt.xlabel('Experience Required')
plt.ylabel('Salary')
plt.legend()
plt.savefig('static/cluster_plot.png')  # save the plot to a file
plt.clf()  # Clear the plot for the second cluster plot

# Generate the second cluster plot and save it to a file
plt.scatter(Y[y_kmeans_loaded_Y == 0, 0], Y[y_kmeans_loaded_Y == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(Y[y_kmeans_loaded_Y == 1, 0], Y[y_kmeans_loaded_Y == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(Y[y_kmeans_loaded_Y == 2, 0], Y[y_kmeans_loaded_Y == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], s=100, c='black', marker='x', label='Centroids')
plt.title('Clusters of Data (Job Title vs Salary)')
plt.xlabel('Job Title')
plt.ylabel('Salary')
plt.legend()
plt.savefig('static/cluster_plot1.png')  # save the plot to a file




# FOR CLUSTERING BASED ON SPECIFIC JOB TITLE
def filter_dataset(csv_file, column, value):
    try:
        df = pd.read_csv(csv_file)
        filtered_df = df[df[column] == value]
        return filtered_df.to_dict('records')
    except Exception as e:
        return str(e)

@app.route('/k_means', methods=['GET', 'POST'])
def k_means():

    # CONNECTED TO "FOR CREATING CLUSTERS AND DISPLAY IMAGE"
    plot_path = 'static/cluster_plot.png'  # Update with the actual path to the generated plot image
    plot_path1 = 'static/cluster_plot1.png'  # Update with the actual path to the generated plot image

    # CONNECTED TO "FOR CLUSTERING BASED ON SPECIFIC JOB TITLE"
    if request.method == 'POST':
        csv_file = 'filled_indeed.csv'
        column = 'Job_Title'
        try:
            filter_value = int(request.form['filter_value'])
            filtered_data = filter_dataset(csv_file, column, filter_value)
            return jsonify({'data': filtered_data})
        except ValueError:
            error_message = "Invalid input. Please enter an integer value."
            return jsonify({'error': error_message})

    # OUTPUT ALL
    return render_template('k_means.html', plot_path=plot_path, plot_path1=plot_path1)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Render the form for user input
        return render_template('predict.html')
    elif request.method == 'POST':
        # Load the data from yellow_pages.json
        with open('YP_Finals.json') as f:
            data = json.load(f)

        reviews = [item['review'] for item in data]

        tokenizer = Tokenizer()

        # Corpus data normalized -> lower, split into strings by new line characters
        c_data = []
        for review in reviews:
            c_data.extend(review.lower().split("\n"))

        # Fit the corpus data to tokens
        tokenizer.fit_on_texts(c_data)

        # Save the tokenizer as a JSON file
        tokenizer_json = tokenizer.to_json()
        with open('tokenizer_jeno.json', 'w', encoding='utf-8') as f:
            json.dump(tokenizer_json, f)

        total_words = len(tokenizer.word_index) + 1

        input_sequences = []

        for line in c_data:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)

        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences,
                                                 maxlen=max_sequence_len,
                                                 padding='pre'))

        # Create predictors and label
        xs, labels = input_sequences[:, :-1], input_sequences[:,-1]

        ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

        # Load the models
        loaded_bi_lstm_model = load_model('bi_lstm_model_jeno.h5')
        loaded_lstm_model = load_model('lstm_model_jeno.h5')
        loaded_gru_model = load_model('gru_model_jeno.h5')

        # Load the tokenizer
# Load the tokenizer from the JSON file
        with open('tokenizer_jeno.json', 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
            tokenizer_config = json.loads(tokenizer_json)
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)


        # Generate predictions
        seed_text = request.form.get('seed_text')
        next_words = int(request.form.get('next_words'))

        # Function to generate predictions
        def generate_predictions(model, tokenizer, seed_text, next_words, max_len):
            generated_text = seed_text
            for _ in range(next_words):
                # Tokenize the seed text
                token_list = tokenizer.texts_to_sequences([seed_text])[0]
                # Pad the token list
                token_list = pad_sequences([token_list], maxlen=max_len, padding='pre')
                # Get the predicted word index
                predicted = np.argmax(model.predict(token_list), axis=-1)
                # Get the predicted word
                output_word = ""
                for word, index in tokenizer.word_index.items():
                    if index == predicted:
                        output_word = word
                        break
                # Append the predicted word to the seed text
                seed_text += " " + output_word
                generated_text += " " + output_word

            return generated_text

        # Generate text predictions using the Bi-LSTM model
        bi_lstm_prediction = generate_predictions(loaded_bi_lstm_model, tokenizer, seed_text, next_words, max_sequence_len - 1)

        # Generate text predictions using the LSTM model
        lstm_prediction = generate_predictions(loaded_lstm_model, tokenizer, seed_text, next_words, max_sequence_len - 1)

        # Generate text predictions using the GRU model
        gru_prediction = generate_predictions(loaded_gru_model, tokenizer, seed_text, next_words, max_sequence_len - 1)

        return jsonify({
            'bi_lstm_prediction': bi_lstm_prediction,
            'lstm_prediction': lstm_prediction,
            'gru_prediction': gru_prediction
        })



#Text Classification

# Load the model
loaded_model = load_model('text_classification_model.h5')

# Load the tokenizer
with open('tokenizer-classify.json', 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

# Compile the loaded model
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

max_length = 100
padding_type = 'post'
trunc_type = 'post'

@app.route('/text_classify', methods=['GET', 'POST'])
def text_classify():
    if request.method == 'POST':
        # Get the input text from the request
        sentence = request.form['text']
        # Tokenize and pad the input text
        sequences = tokenizer.texts_to_sequences([sentence])
        padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        # Make predictions
        predictions = loaded_model.predict(padded)
        # Convert probabilities to class labels
        threshold = 0.5
        labels = ['positive' if p > threshold else 'negative' for p in predictions]
        # Return the prediction result as JSON
        return jsonify(sentence=sentence, label=labels[0])
    else:
        return render_template('text_classify.html')



if __name__ == '__main__':
    app.run(debug=True)




