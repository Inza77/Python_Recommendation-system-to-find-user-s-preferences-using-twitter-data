# Python_Recommendation-system-to-find-user-s-preferences-using-twitter-data
Creating a Recommendation system using python for capstone project Biztravel
Recommendation system is applied in the creation of capstone project web application called Biztravel. It is an AI-driven decision support system designed to optimize the leisure time of business travelers in unfamiliar destinations.

Data Acquisition:
We collect user data and user-liked comment author data from the Twitter API. This includes information about user interactions, such as likes and comments, to understand their preferences and interests.

Sentiment Analysis:
Using Google NLP, we analyze the sentiment of user-generated content. Positive sentiment analysis helps us identify content that users enjoy and find valuable. This step is crucial for understanding user preferences and categorizing content effectively.

Content Categorization:
Based on the sentiment analysis results, we categorize the content into different categories or topics. This segmentation enables us to group similar types of content together and create a more personalized recommendation experience for users.

Model Validation and Deployment
Cosine Similarity:
Cosine similarity is utilized to measure the similarity between user preferences and potential places to recommend. It is a metric commonly used in recommendation systems to compare the preferences of different users or items. By calculating the cosine similarity between user profiles and place profiles, we can identify places that are most like what the user enjoys.

Recommendations:
Using the computed cosine similarities, we generate personalized recommendations for users. These recommendations are tailored to each user's preferences and are based on the places that have the highest similarity to the content they have previously interacted with positively on Twitter.

import os

# Replace "/User/your-service-account-key.json" with the actual path to your service account key JSON file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/inzamamahamedmoideen/Downloads/thermal-proton-415501-08d96de857f3.json"

# Now you can use Google Cloud client libraries, and they will use the credentials from the specified JSON file

import argparse
import json
import os
from google.cloud import language_v1
import numpy
import tweepy




def content(contentc):
    list_of_categories=[]
    client = language_v1.LanguageServiceClient()

    text_content = contentc
    type_ = language_v1.Document.Type.PLAIN_TEXT
    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}
    encoding_type = language_v1.EncodingType.UTF8

    content_categories_version = (
        language_v1.ClassificationModelOptions.V2Model.ContentCategoriesVersion.V2)

    # Analyze sentiment
    response1 = client.analyze_sentiment(
        request={"document": document, "encoding_type": encoding_type}
    )

    # Analyze content categories
    response2 = client.classify_text(
        request={
            "document": document,
            "classification_model_options": {
                "v2_model": {"content_categories_version": content_categories_version}
            }
        }
    )

    a = set()

    for sentence in response1.sentences:
        if sentence.sentiment.score > 0.2:
            #print(f"Sentence text: {sentence.text.content}")
#             print(f"Sentence sentiment score: {sentence.sentiment.score}")
            #print("//////////////")
            #print(response2)
            #print("//////////////")
            for category in response2.categories:
                if category.confidence > 0.2:
                    category_name = category.name
                    list_of_categories.append(category_name)
                    
    
    return list_of_categories

# Get your Twitter API credentials and enter them here
consumer_key = "NxDyN9wjMoEotCsu0kvaGM6Qv"
consumer_secret = "hq61mYCUmjZjdj7B79VdO9qLzfNX3yAbB80Upau7clDvcXqnJz"
access_key = "2213402124-f9faofdwJXEJT84SAHttpRwXv1KA4XQwW0ZF7wH"
access_secret = "yCbbcjzh2ib5lppqt5aY5xjHXV1eStMgUOg424Gg8TNqb"
bearer_token = "AAAAAAAAAAAAAAAAAAAAACFOsQEAAAAAEQUAfEmQTSOFV75DtYLWhtz1L6I%3DbAoXPA9C4cow1qYoGoqG0o0WxapA8ZRO5QBcsK2l63j5EXO6sc"

def get_user_tweets_analyzed(username):
    # Tokens
    api = tweepy.Client(
        bearer_token=bearer_token,
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_key,
        access_token_secret=access_secret
    )

    user = api.get_user(username=username)
    user_id = user.data.id

    tweets = api.get_users_tweets(id=user_id, max_results=5)
    all_categories = get_categories_for_tweets(tweets)
    list_of_tuple_username_categories = [(username, i) for i in all_categories]
    return list_of_tuple_username_categories
    print("Username")
   
    
def deep_flatten(lst):
    flattened_list = []
    for i in lst:
        if isinstance(i, list):
            flattened_list.extend(deep_flatten(i))
        else:
            flattened_list.append(i)
    return flattened_list

def get_categories_for_tweets(tweets):
    total_categories =[]
    for i in range(len(tweets.data)):
         tweet_content = tweets.data[i].text
         categories = content(tweet_content)
         total_categories.append(categories)
    return deep_flatten(total_categories)

        
def get_main_user_tweets(username):
    # Tokens
    global_result = []
    main_user_result = get_user_tweets_analyzed(username)
    #print("Main user result ")
    global_result.append(main_user_result)
    #print(main_user_result)
        
    liked_tweets_result=get_liked_tweets_for(username)
    global_result.append(liked_tweets_result)
    return deep_flatten(global_result)

def get_liked_tweets_for(username):
    function_result=[]
    api = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=consumer_key,
    consumer_secret=consumer_secret,
    access_token=access_key,
    access_token_secret=access_secret)    
    user= api.get_user(username=username)
    #print(user)    
    user_id= user.data.id
    #print(user[0])
    response = api.get_liked_tweets(id=user_id, expansions='author_id',max_results=5)
#     print(response.includes)
    for i in range(len(response.includes["users"])):
        #print(response.includes["users"][i].username)
        liked_user_result=get_user_tweets_analyzed(response.includes["users"][i].username)
        #print("Liked user result ",response.includes["users"][i].username)
        function_result.append(liked_user_result)
        #print(liked_user_result)
    return function_result

# print("Enter one username")
# user = input("Enter username:")
# df_input= get_main_user_tweets(user)

# print("******************")
# print(df_input)

import pandas as pd
def create_dataframe(df_input):
    # Process the categories to extract the first part and tally counts
    processed_data = []
    for user, category in df_input:
    # Extract the main category
        main_category = category.split('/')[1]  # Split by '/' and take the second element
        processed_data.append((user, main_category))

# Convert processed data into a DataFrame
    df_processed = pd.DataFrame(processed_data, columns=['User', 'Category'])

# Create a pivot table with users as rows, categories as columns, and counts as values
    pivot_df = pd.pivot_table(df_processed, index='User', columns='Category', aggfunc=len, fill_value=0)
    return pivot_df

# df=create_dataframe(df_input)
# print(df.head())

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
def create_recommendation(df):
    # Convert the DataFrame into a sparse matrix for efficiency
    sparse_df = sparse.csr_matrix(df.values)

    # Calculate the cosine similarity matrix
    cosine_sim = cosine_similarity(sparse_df, dense_output=False)

    # Convert the cosine similarity matrix to a DataFrame for easier manipulation
    cosine_sim_df = pd.DataFrame(cosine_sim.toarray(), index=df.index, columns=df.index)

    # Get the similarity values for the first user (Alice) with all users
    similarity_scores = cosine_sim_df.iloc[0]

    # Sort the users based on similarity scores, excluding the first user herself
    sorted_users = similarity_scores.sort_values(ascending=False)[1:]

    # Identify the top N similar users for recommendations
    top_n_users = sorted_users.head(3).index

    # Compile recommendations from top N similar users
    recommendations = pd.Series(dtype='float64')
    for user in top_n_users:
        # Get categories that the first user hasn't interacted with yet (rated 0)
        unseen_categories = df.loc['CapstoneP003'][df.loc['CapstoneP003'] == 0].index
        # Add the unseen categories rated by the similar user to the recommendations
        recommendations=pd.concat([recommendations, df.loc[user, unseen_categories]])
        #recommendations = recommendations.append(df.loc[user, unseen_categories])

    # Group recommendations by category and calculate the average rating from the top N users
    recommendations = recommendations.groupby(recommendations.index).mean().sort_values(ascending=False)
    print(f"Recommended categories for CapstoneP003 based on top similar users' preferences: {recommendations.index.tolist()}")
    return recommendations
    
#FLASK API

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import argparse
import json
import os
from google.cloud import language_v1
import numpy
import tweepy

app = Flask(__name__)

# Define the API route
@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    # Get the username from the request
    username = request.json['username']
    df_input= get_main_user_tweets(username)
    print(df_input)
    df=create_dataframe(df_input)
    print(df.head())
    # Call the function to get the recommendations for the username
    user_recommendations = create_recommendation(df)
    print(user_recommendations.head())
    # Convert the recommendations to a JSON response
    response = {'username': username, 'recommendations': user_recommendations.index.tolist()}
    return jsonify(response)

if __name__ == '__main__':
    app.run()

