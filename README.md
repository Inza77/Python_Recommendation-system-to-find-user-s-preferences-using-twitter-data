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

![image](https://github.com/Inza77/Python_Recommendation-system-to-find-user-s-preferences-using-twitter-data/assets/167274893/049d4187-16b2-4789-b023-b6b85a974fc3)
