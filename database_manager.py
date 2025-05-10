import pandas as pd
import os
import random

class UserManager:
    def __init__(self, user_database_path='user_database.csv'):
        self.user_database_path = user_database_path

        if os.path.exists(self.user_database_path):
            self.user_df = pd.read_csv(self.user_database_path)
        else:
            self.user_df = pd.DataFrame(columns=[
                'Username',
                'Reviewer History',
                'Upvotes'
            ])
            self.user_df.to_csv(self.user_database_path, index=False)

    def save_database(self):
        self.user_df.to_csv(self.user_database_path, index=False)

    def user_exists(self, username):
        return username in self.user_df['Username'].values

    def get_user_features(self, username):
        if self.user_exists(username):
            user_row = self.user_df[self.user_df['Username'] == username].iloc[0]
            hist = user_row["Reviewer History"]
            if user_row["Reviewer History"] == "New":
                hist = 0
                user_row["Reviewer History"] = "Old"
                # Update the database
                self.user_df[self.user_df["Username"] == username] = user_row
                self.save_database()
            else:
                hist = 1
            return {
                "Upvotes": int(user_row["Upvotes"]),
                "ReviewerHistory": hist,
            }
        else:
            return None

    def create_new_user(self, username):
        new_user = {
            "Username": username.lower(),
            "Upvotes": random.randint(0, 110) if random.choice([0,1]) else 0,  # Randomly assign upvotes for new users for PoC
            "Reviewer History": "New",
        }
        new_df = pd.DataFrame([new_user])
        self.user_df = pd.concat([self.user_df, new_df], ignore_index=True)
        self.save_database()
        print(f"User '{username.lower()}' created successfully.")

    def lookup_or_create_user(self, username):
        features = self.get_user_features(username.lower())
        if features is not None:
            print(f" User '{username.lower()}' found. Features retrieved.")
        else:
            print(f" New user '{username.lower()}' detected. Creating profile...")
            self.create_new_user(username)
            features = self.get_user_features(username.lower())
        return features

