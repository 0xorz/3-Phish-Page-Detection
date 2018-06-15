# Phish-Page-Detection

```

   _____                   _   _____  _     _     _
  / ____|                 | | |  __ \| |   (_)   | |
 | (___   __ _ _   _  __ _| |_| |__) | |__  _ ___| |__
  \___ \ / _` | | | |/ _` | __|  ___/| '_ \| / __| '_ \
  ____) | (_| | |_| | (_| | |_| |    | | | | \__ \ | | | - Detection
 |_____/ \__, |\__,_|\__,_|\__|_|    |_| |_|_|___/_| |_|
            | |
            |_|

```

Welcome to SquatPhish-phishing-detection!

SquatPhish-phishing-detection is part of SquatPhish project to detect general phishing pages.

A machine learning model to identify phishing pages by looking at:

* HTML text - searching for brand name and signin keywords in HTML source code
* HTML structure - searching for submission forms and their attributes
* IMAGE text - searching for texts directly from image

We apply tesseract (a Deep learning based OCR engine) to extract texts from images.

We also NLP analysis to filter and clean nonsense words.

It supports:

- [x] Directly detection of potential phishing pages
- [x] A behavior-based model to investigate general phishing behaviors
- [x] A machine-learning-based (RandomForest) to combine all the properties to make a final decision


## Install OCR, NLTK and ML dependences
```
bash install.sh
```

## Demo

Run the demo to get predictions of testing samples under test folder.

1 is predicted as a phsihing page and 0 is predicted as a benign page.

```
python3 demo.py

Prediction result:	 googlw.lt----1.0----[0.28993355726427317, 0.7100664427357269]

Prediction result:	 googlw.cl----1.0----[0.09047988302948216, 0.9095201169705178]

Prediction result:	 googlw.co.il----1.0----[0.24020989615461588, 0.7597901038453841]

Prediction result:	 goofgle.se----1.0----[0.31088049798443707, 0.6891195020155627]

Prediction result:	 facebook-c.com----1.0----[0.22163562365428768, 0.7783643763457124]

Prediction result:	 faceb00k.bid----1.0----[0.17656695383979332, 0.8234330461602067]

Prediction result:	 sewauk.org----1.0----[0.25669347640761164, 0.7433065235923882]

Prediction result:	 100022538-facebook.com----0.0----[0.9193098619543378, 0.08069013804566202]

```

Get feature vectors.

```
python3 feature_extract.py
```

## API

We provide a prediction API.
```                                                                                                                                                               ketian@ketian
usage: predict.py [-h] [-t HTML] [-i IMG]

running analysis...

optional arguments:
  -h, --help            show this help message and exit
  -t HTML, --html HTML  A html source data to extract features
  -i IMG, --img IMG     A image data to extract features

```

Example:

```
python3 predict.py  --img=./test/facebook-c.com..screen.png
                    --html=./test/facebook-c.com..source.txt
```


## Disclaimer and Reference

This is a research prototype, use at your own risk.

If you feel this tool is useful, cite the tool as :dog2: SquatPhish :dog2: is highly appreiciated.

## Acknowledgement

Core contributor: ke tian @ririhedou

Thanks hang hu @0xorz for reproduction testing.

Current version is 0.0.2, updated at June 14 2018