
# Sentiment Analysis Software

This is a simple website that allows users to input either a YouTube video URL or a sentence and get a binary sentiment classification prediction based on a pre-trained deep learning model. The model was built using TensorFlow and Keras, and uses a Bidirectional LSTM architecture with an embedding layer.

## Tech Stack

- Tech Stack
- Python
- Flask
- HTML/CSS
- TensorFlow
- Keras
- YouTube API

## Installation

To run the website, you will need to have Python 3 installed. You can clone this repository using the following command:

```bash
git clone https://github.com/AbhiSinghiitr/sentiment-analysis-software.git
```

Next, navigate to the cloned directory and install the required Python packages using pip:

```bash
cd sentiment-analysis-software
pip install -r requirements.txt
```

## Usage

To run the website, simply run the app.py file using Python:

```bash
python app.py
```

This will start the website on http://127.0.0.1:3000. You can open this URL in your web browser to access the website.

Once the website is loaded, you can enter some text in the input box and click the "Submit" button to get a binary classification prediction. The prediction is displayed on the page along with the probability score.

### Predicting sentiment for a sentence

To predict the sentiment for a sentence, simply enter the sentence in the input box and click the "Submit" button. The prediction is displayed on the page along with the probability score.

### Scraping YouTube comments

To scrape comments from a YouTube video, enter the YouTube video URL in the input box and click the "Scrape" button. The website will scrape the comments from the video and store them in a .csv file. You can then click the "Submit" button to run the pre-trained model on the comments data and get a binary sentiment classification prediction. The prediction is displayed on the page along with the number of positive and negative comments.

## How it works

The website uses a pre-trained deep learning model to make binary sentiment classification predictions on text data. The model is based on a Bidirectional LSTM architecture, which allows the model to process the text in both forward and backward directions, capturing both the past and future context of the text.

The model is trained using a binary cross-entropy loss function and optimized using the Adam optimizer. The input text is first processed using an embedding layer, which converts each word into a vector representation. The embedding layer is followed by two Bidirectional LSTM layers, which process the text data in both directions. The output of the second LSTM layer is then passed through a Dense layer with a ReLU activation function, followed by a Dropout layer with a dropout rate of 0.5. Finally, the output is passed through a Dense layer with a sigmoid activation function, which produces a binary sentiment classification prediction.

## Customization

If you want to use your own pre-trained model, you can replace the model.keras file in the models directory with your own model file. Make sure the model file is in the Keras format (*.keras) and has the same architecture as the provided model.

## Features
- Can handle two types of inputs: YouTube video URL or a sentence
- Scrapes comments from YouTube video using YouTube API and stores them in a CSV file
- Performs sentiment analysis on the comments stored in the CSV file
- Accepts a sentence as input and performs sentiment analysis on it
- Displays the percentage of positive and negative comments or sentence
- User-friendly web interface

## Roadmap

1. Built the basic Flask app with a form that allows the user to  input a YouTube URL or a sentence
2. Integrated the YouTube Data API to scrape comments from the given video URL and store them in a CSV file
3. Developed a deep learning model using TensorFlow and Keras to perform sentiment analysis on the comments stored in the CSV file
4. Integrated the model with the Flask app to display the percentage of positive and negative comments or sentence
5. Added styling to the web interface using HTML/CSS

## Future Scope

- Automated Spam Detection
- Better insight on comments through replies
- Better analysis of confusing comments such as sarcastic comments
- Sentiment analysis for multiple languages

## Credits

This website was developed as part of a project for the course CSN254. The pre-trained model was developed using the Sentiment140 dataset, which can be found at https://www.kaggle.com/datasets/kazanova/sentiment140 and IMDB Movie Rating dataset, which can be found at https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis


## Authors

- [Sarish Nilakhe](https://github.com/Shinchan9913)
- [Abhishek Kumar Singh](https://github.com/AbhiSinghiitr)
- [Shambhoolal Narwaria](https://github.com/narwariasmb)
- [Alhan Charan Beshra](https://github.com/ezio2605)
- [Abhishek Raj](https://github.com/Abhi9708bittu)
- [Aryan Batra](https://github.com/Batraji7)

