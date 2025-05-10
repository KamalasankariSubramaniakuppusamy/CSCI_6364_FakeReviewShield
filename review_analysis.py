# review_analyzer.py
import pandas as pd

class ReviewAnalyzer:
    def __init__(self, user_features):
        self.user_features = user_features
        pass

    def analyze_review(self, review_text, rating):
        user_row = self.user_features

        # 1. Review Length
        review_length = len(review_text.split())
        #Classify review length
        if review_length < 10:
            review_length_class = 0
        elif review_length < 20:
            review_length_class = 1
        else:
            review_length_class = 2


        #2. Incentives
        incentive_keywords = ["free", "gift", "reward", "bonus", "promotion","promo","sponsor","sponsored","testing","trial","sample","survey","contest","raffle","sweepstakes","for free","for review","exchange for a review","incentive","paid promotion"]
        incentives = 1 if any(keyword in review_text.lower() for keyword in incentive_keywords) else 0

        #3. reviewer history
        # Heuristic: If the user has a history of writing reviews
        # Get reviewer history from user row
        reviewer_history = int(user_row["ReviewerHistory"])

        #4. Upvotes
        # Get upvote count from user row
        upvote_count = int(user_row["Upvotes"])

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


        #5. Photo/Video
        # Get photo/video presence from user row
        photo_video = 0 # Default to 0 (No) since we are making CLI for PoC


        #6. Ratings
        # Get ratings from user row
        ratings = float(rating)


        # Final feature list
        features =[
            ratings,
            photo_video,
            reviewer_history,
            review_length_class,
            incentives,
            upvote_class,
            ]

        return features
