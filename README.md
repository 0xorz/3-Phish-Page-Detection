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
```
python3 demo.py
```



# API




## Disclaimer

research prototype, use at your own risk.