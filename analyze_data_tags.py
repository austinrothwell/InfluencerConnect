import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# Directory that metadata is saved in
metadata_directory = './users/'

# Load Instagram user list
instagram_users = open('instagram_users_organized.txt').read().split('\n')

user_str = ''
global_str = []

# Loop over users
for username in instagram_users:

    metadata = json.load(open(metadata_directory + username + '.json'))

    for i in range(1, len(metadata)):
        if not metadata[i]['is_video']:
            user_str += ' ' + metadata[i]['image_contents'][0]
            user_str += ' ' + metadata[i]['image_contents'][1]
            user_str += ' ' + metadata[i]['image_contents'][2]
            # for j in range(0, len(metadata[i]['tags'])):
            #     user_str += ' ' + metadata[i]['tags'][j]

    global_str.append(user_str)
    user_str = ''

# Vectorize image content strings
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(global_str)

# Fit K-Nearest Neighbors model
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(X)

# Specify target brand profile position in 'instagram_user' list
target = 25
X_tilde = X[target]
print('Target profile is:')
print(instagram_users[target])

# Find nearest neighbors
neighbors = neigh.kneighbors(X_tilde)[1][0]
print('\n' + 'Most closely related profiles are:')
for j in range(1, len(neighbors)):
    print(instagram_users[neighbors[j]])

# Multi-dimensional Scaling (MDS) to visualize user distance in 2D
X_lowdim = MDS(n_components=2).fit_transform(X.toarray())
ax = plt.gca()
c1 = ax.scatter(X_lowdim[0:5, 0], X_lowdim[0:5, 1], label='dogs')
c2 = ax.scatter(X_lowdim[5:10, 0], X_lowdim[5:10, 1], label='mountains')
c3 = ax.scatter(X_lowdim[10:15, 0], X_lowdim[10:15, 1], label='pizza')
c4 = ax.scatter(X_lowdim[15:20, 0], X_lowdim[15:20, 1], label='cats')
c5 = ax.scatter(X_lowdim[20:25, 0], X_lowdim[20:25, 1], label='cars')
c6 = ax.scatter(X_lowdim[target, 0], X_lowdim[target, 1], label='target', marker='*', color='y')
plt.legend(handles=[c1, c2, c3, c4, c5, c6])
plt.title('Target brand profile: ' + instagram_users[target])

# Annotate data points
for i in range(0, 25):
    plt.annotate(instagram_users[i], xy=(X_lowdim[i, 0], X_lowdim[i, 1]))

plt.show()



