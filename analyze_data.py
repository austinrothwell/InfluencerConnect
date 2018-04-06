import json
import numpy as np

# Directory that metadata is saved in
metadata_directory = '/Users/Taylor/PycharmProjects/InfluencerConnect/users/'

# Load Instagram user list
instagram_users = open('instagram_users2.txt').read().split('\n')

features = []

# Loop over users
for user, username in enumerate(instagram_users):

    metadata = json.load(open(metadata_directory + username + '.json'))

    for i in range(0, len(metadata)):
        features.append(metadata[i]["image_contents"][0])

print(features)