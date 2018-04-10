import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import re
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # To silence some TensorFlow warnings

# Load Instagram user list
instagram_users = open('instagram_users.txt').read().split('\n')

for username in instagram_users:

    if not os.path.isfile('./users/' + username + '.json'):
        n_posts = 100   # Total number of posts to scrape (images and videos)
        n_images = 50   # Number of images you want to analyze

        # Scrape Instagram profile for n_images most recent posts and collect metadata
        os.system('instagram-scraper ' + username + ' --maximum ' + str(n_posts) +
                  ' -u austinrothwell -p ------- --media-metadata --destination ./users/' + username)

        # Directory that user images are saved in
        images_directory = './users/' + username + '/'

        # Load user metadata, remove videos, remove video metadata from metadata
        metadata = json.load(open(images_directory + username + '.json'))
        os.system('rm ' + images_directory + '*.mp4')
        [metadata.remove(post) for post in metadata if post['is_video']]

        metadata = metadata[0:n_images]  # Trim metadata to size of n_images

        # Classify each image and show with metadata
        for image in range(0, n_images):
            image_name = metadata[image]['urls'][0].split('/')[-1:][0]
            filename = os.fsdecode(image_name)
            if filename.endswith(".jpg"):

                # # Resize image to 299x299 (the size inception-v3 was trained on) -- Not sure if this helps
                # im = cv2.imread(images_directory + filename)
                # im_resize = cv2.resize(im, (299, 299))
                # cv2.imwrite(images_directory + filename.split('.')[0] + '_resize.jpg', im_resize)

                # Image classification
                out = subprocess.check_output('python ./models/tutorials/image/imagenet/classify_image.py --image_file '
                                              + images_directory + filename, shell=True)

                # Get image metadata
                image_class = str(out).split('\\')[0][2:] + '\n' + str(out).split('\\')[1][1:]
                n_likes = metadata[image]['edge_media_preview_like']['count']
                n_comments = metadata[image]['edge_media_to_comment']['count']
                try:
                    caption = metadata[image]['edge_media_to_caption']['edges'][0]['node']['text']
                    tags = ' '.join(metadata[image]['tags'])
                except:
                    caption = 'n/a'
                    tags = 'n/a'

                # Write image contents to user JSON file
                metadata[image]['image_contents'] = [s.split(' (')[0] for s in str(out)[2:-3].split('\\n')]
                metadata[image]['image_scores'] = [float(re.findall("\d+\.\d+", s)[0]) for s in str(out)[2:-3].split('\\n')]
                json.dump(metadata, open(images_directory + username + '.json', 'w'),
                          sort_keys=True, indent=4, separators=(',', ': '))

                # # Show image and predicted labels
                # img = mpimg.imread(images_directory + filename)
                # imgplot = plt.imshow(img)
                # plt.title(image_class)
                # plt.xlabel('likes: ' + str(n_likes) + ', comments: ' + str(n_comments) + '\n' +
                #            'caption: ' + caption + '\n' + 'tags: ' + tags)
                # plt.gcf().subplots_adjust(bottom=0.18)
                # plt.show()

        # Delete all images from user directory and move .json metadata to .\users
        os.rename(images_directory + username + '.json', './users/' + username + '.json')
        os.system('rm ' + images_directory + '*.jpg')
        os.rmdir(images_directory)


