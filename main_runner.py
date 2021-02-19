import torch
import joblib
import torch.nn as nn
import numpy as np
from PIL import Image
import argparse
from torchvision import models

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img', default='im-received.jpg', type=str,
    help='path for the image to test on')
args = vars(parser.parse_args())

# load label binarizer
# this is just the working directory.
lb = joblib.load('/lb.pkl')

'''MODEL'''
def model(pretrained, requires_grad):
    model = models.resnet50(progress=True, pretrained=pretrained)
    # freeze hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification layers learnable
    model.fc = nn.Linear(2048, len(lb.classes_))
    return model
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model(pretrained=False, requires_grad=False).to(device)
# this is just the working directory.
model.load_state_dict(torch.load('/home/johnorourke/PycharmProjects/Stuff/rockdog/output/model.pth'))
print('Model loaded')

# this is just the working directory.
image = Image.open(f"/home/johnorourke/johnorourke/PycharmProjects/Stuff/rockdog/{args['img']}")
image_copy = image.copy()
image = np.transpose(image, (2, 0, 1)).astype(np.float32)
image = torch.tensor(image, dtype=torch.float).to(device)
image = image.unsqueeze(0)
print(image.shape)
outputs = model(image)
_, preds = torch.max(outputs.data, 1)
print(f"Predicted output: {lb.classes_[preds]}")

def answer_return():
    if lb.classes_[preds]=='rock':
        return 'rock'
    else:
        return 'not a rock'


#cv.putText(image_copy, lb.classes_[preds], (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
image_copy.show()
#cv.imwrite(f"/home/johnorourke/Downloads/output/{args['img']}.jpg", image_copy)
#cv.waitKey(0)