"# Machine-Learning-Tweet-Predicting" 

1. To run this code you need to have access to internet so that it can pull data from the Twitter API
2. When prompted to do so, the program will ask the user to input a Twitter username
3. After inputting the Twitter username, the program will ask the user to input a category to check the classification algorithm against. The possible categories are one of the following
    1. Politics
    2. Sports
    3. Tech
    4. Music
    5. TV

4. The program will return a certain number of tweets and will classify each of those tweets individually. With the percent accuracy of the overall classification of those tweets to a particular category.

5. There are a few variables within the program that you can edit to change the number of results that you obtain, to change the number of training data that you get, and to change the number of epochs that the program will run. 
    1. To change the number of training tweets checked, you must change the count variable in t.search(), within the                tag_to_tweet() function
    2. To change the number of epochs, you must change the epochs variable within the train() function
    3. To change the number of results that you obtain, you must change the tweetCount variable within the classify_user()          function
