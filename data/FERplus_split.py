import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='./FERPlus', type=str, help='image save path')
    parser.add_argument('--data_path', default='./fer2013/fer2013.csv', type=str, help='data csv path')
    parser.add_argument('--label_path', default='./fer2013new.csv', type=str, help='label csv path')
    
    args, unparsed = parser.parse_known_args()
    return args

def main():
    data = pd.read_csv(args.data_path)
    labels = pd.read_csv(args.label_path)
    # extract new emotions
    emotions = labels.drop(columns=['Usage','Image name'])
    emotion_map = {
        'neutral':0,
        'happiness':1,
        'surprise':2,
        'sadness':3,
        'anger':4,
        'disgust':5,
        'fear':6,
        'contempt':7,
        'unknown':8,
        'NF':9
    }
    new_emotion = emotions.idxmax(axis = 1).map(emotion_map).astype('int32')
    # generate new data table
    # | emotion | pixels | Usage | Image name |
    new_data = data
    new_data['Usage'] = labels['Usage']
    new_data['emotion'] = new_emotion
    new_data['Image name'] = labels['Image name']
    process_data(new_data, args.save_path)

def process_data(data, path):
    import os
    import matplotlib.pyplot as plt
    count = 0
    '''
    Args:
    data (DataFrame): df of face images
        columns:
            emotion
            pixels
            Usage
            Image name
    '''
    n = len(data) # num of examples
    for i in range(n):
        pixels = data['pixels'][i]
        img = np.fromstring(pixels, dtype='int', sep=' ')
        img = img.reshape(48, 48)
        usage = data['Usage'][i]
        emotion = data['emotion'][i]
        if emotion > 7:
            continue
        name = data['Image name'][i].replace('.png','.jpg')
        folder = os.path.join(path, usage, str(emotion))
        if not os.path.exists(folder):
            os.makedirs(folder)
        img_path = os.path.join(folder, name)
        count +=1
        print(count, img_path)
        plt.imsave(img_path, img, cmap='gray')

if __name__ == '__main__':
    args = get_parser()
    main()
