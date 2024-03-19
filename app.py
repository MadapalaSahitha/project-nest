from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import hashlib
import pickle
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['instagram']  
user_collection = db['users']

# Load the saved model
with open('instagram_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Preprocessing function for X1
def preprocess_data_X1(profile_pic, nums_length_username, fullname_words, nums_length_fullname, name_username, description_length, external_URL, private, num_posts, num_followers, num_follows):
    data = [profile_pic, nums_length_username, fullname_words, nums_length_fullname, name_username, description_length, external_URL, private, num_posts, num_followers, num_follows]
    processed_data = np.array([data])
    return processed_data

# Route for home page
@app.route('/')
def home():
    return render_template('login.html')

# Route for signup page
# Route for signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirmation_password = request.form['confirmation_password']
        
        # Check if passwords match
        if password != confirmation_password:
            return render_template('signup.html', message="Passwords do not match. Please try again.")
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        user_exists = user_collection.find_one({'username': username})
        if user_exists:
            return render_template('signup.html', message="Username already exists. Please choose another one.")
        else:
            user_collection.insert_one({'username': username, 'password': hashed_password})
            # Redirect to the login page after successful signup
            return redirect(url_for('login'))
    return render_template('signup.html')




# Route for login page
# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        # Check if the user exists in the database
        user = user_collection.find_one({'username': username})
        if user:
            # Compare the hashed password with the password stored in the database
            if user['password'] == hashed_password:
                return redirect(url_for('index', username=username))
            else:
                return render_template('login.html', message="Invalid password. Please try again.")
        else:
            return render_template('login.html', message="User does not exist. Please sign up.")
    return render_template('login.html')



# Route for index page
@app.route('/index/<username>')
def index(username):
    return render_template('index.html', username=username)

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    profile_pic = float(request.form['profile_pic'])
    nums_length_username = float(request.form['nums_length_username'])
    fullname_words = float(request.form['fullname_words'])
    nums_length_fullname = float(request.form['nums_length_fullname'])
    name_username = float(request.form['name_username'])
    description_length = float(request.form['description_length'])
    external_URL = float(request.form['external_URL'])
    private = float(request.form['private'])
    num_posts = float(request.form['num_posts'])
    num_followers = float(request.form['num_followers'])
    num_follows = float(request.form['num_follows'])

    # Preprocess the input data
    processed_data_X1 = preprocess_data_X1(profile_pic, nums_length_username, fullname_words, nums_length_fullname, name_username, description_length, external_URL, private, num_posts, num_followers, num_follows)

    # Check the number of features used for training the scaler
    num_features_scaler = scaler.mean_.shape[0]

    # Ensure that the number of features used for training the scaler matches the expected number of features
    expected_num_features = 11
    if num_features_scaler != expected_num_features:
        raise ValueError(f"Number of features used for fitting the scaler ({num_features_scaler}) does not match the expected number of features ({expected_num_features})")

    # Scale the input data for X1
    X1_scaled = scaler.transform(processed_data_X1)

    # Prediction
    prediction = model.predict(X1_scaled)

    # Choose the class with the highest probability
    result = "Real" if prediction[0] == 1 else "Fake"

    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
