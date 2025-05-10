import pandas as pd
import random

# Load the dataset
file_path = "DATASET.csv"  # Replace with the actual path to your file
df = pd.read_csv(file_path)

def rev_update(row):
        review_length = row
        #Classify review length
        if review_length < 10:
            review_length_class = 0
        elif review_length < 20:
            review_length_class = 1
        else:
            review_length_class = 2
        
        return review_length_class

def get_incentives(row):
    incentive_keywords = ["free", "gift", "reward", "bonus", "promotion","promo","sponsor","sponsored","testing","trial","sample","survey","contest","raffle","sweepstakes","for free","for review","exchange for a review","incentive","paid promotion"]
    incentives = 1 if any(keyword in row.lower() for keyword in incentive_keywords) else 0
    return incentives

def upvote_class(row):
    upvote_count = row
    # Classify upvote count into categories
    if upvote_count < 10:
        upvote_class = 0  # Low
    elif upvote_count < 25:
        upvote_class = 1
    elif upvote_count < 50:
        upvote_class = 2
    elif upvote_count < 100:
        upvote_class = 3
    else:
        upvote_class = 4  # High
    return upvote_class

df["ReviewLengthClass"] = df["ReviewLength"].apply(rev_update)
df["IncentivesClass"] = df["Review"].apply(get_incentives)
df["UpvotesClass"] = df["Upvotes"].apply(upvote_class)

print(df["ReviewLengthClass"].value_counts())
print(df["IncentivesClass"].value_counts())
print(df["UpvotesClass"].value_counts())

df.drop(columns=["ReviewLength", "Incentives", "Upvotes"], inplace=True)

# Save the updated DataFrame to a new CSV file
df.to_csv("DATASET_UPDATED.csv", index=False)