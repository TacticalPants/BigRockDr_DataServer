# BigRockDr_DataServer

# Big Rock Doctor - Data Server

This is part 1 of a 2 part project.  This contains the server and the data analysis model.
Part 2 contains the user interface.

## Goal:
- Create a Neural Network that can determine if the primary focus of a photograph is a rock.
- Create a server for the Neural Network to opperate on.
- Create an API to accept a photograph from a corrisopnding application and serve it to the Nerual Network. The API then sends a response from the network back to the application.

## Currently:
-  The server API accepts a photo send from an application that is in the network.  I have not allow this to operate in the wild.
-  The Neural Network uses a database from Kaggle that is called natural images and about 500 pictures of rocks that I have added to the collection.
-  The network is biased to thinking everyting is a dog or 'not a rock', it will require tweaking in order to become more accurate.

## Dependencies 
Python

### Libraries
- Flask
- PIL
- Sklearn
- Pandas
- Numpy
- Joblib
- Torch
- Torchvision
- imutils
- albumentations
- tdqm

## To Do:
- Modify the model to opperate more accurately
- Modify API to send response to the application
- Launch flask server so that it can operate in the wild.
