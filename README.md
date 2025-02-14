# Dog Breed Classification using InceptionV3 with 96% Accuracy

<p align="center">
  <strong>Accuracy: 0.9614</strong><br>
  <strong>Precision: 0.9669</strong><br>
  <strong>Recall: 0.9614</strong><br>
  <strong>F1-score: 0.9589</strong>
</p>


Hey there! 👋 This project is still in the works. Currently, you'll find only the model training and basic web pages for image uploading and result display; I am still looking for a way to host the web app (I tried PythonAnywhere, but it encountered compatibility issues. I also tried Ngrok but the generated link is temporary 😅).

Stay tuned—things are still evolving!

## Test image:

I've tested on a random image of a cocker dog which I found on pinterest, the top 1 result is indeed cocker with 0.99 confidence score.

<img width="481" alt="uploaded" src="https://github.com/user-attachments/assets/09a9bc61-b474-458b-ae17-5101d3c90d65">

<img width="1043" alt="result" src="https://github.com/user-attachments/assets/2317cae5-ac42-49ac-837a-20e62bd64a17">

## Simple web interface:
<img src="https://github.com/user-attachments/assets/712dc70e-b9a2-4dc0-8372-1f9d4687e3d2" width="400" />
<img src="https://github.com/user-attachments/assets/4c6ab2ae-4812-4de2-9088-49791e092819" width="400" />

## Note:
The bulldog class has the lowest accuracy because the training set contains only English bulldogs, while the testing set includes both English and French bulldogs. Later I'll try to enhance the dataset.

## Steps of running python web app (in local environment):

1. go to your terminal/cmd and locate in the directory that contains 'app3.py';

2. run command 'python app3.py' or 'python3 app3.py' to start the web app;

3. use the local URL to access the web pages


## Reference:

Dataset used: https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set/data

Test image is from: https://www.pinterest.com/pin/4292562138152707/

