# Spatial Validations HERE Hackathon
[![HTML](https://img.shields.io/badge/HTML-%23E34F26.svg?logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/HTML)
[![CSS](https://img.shields.io/badge/CSS-639?logo=css&logoColor=fff)](https://developer.mozilla.org/en-US/docs/Web/CSS)
[![Javascript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=fff)](https://keras.io/)

Our 2nd place submission for the Spatial Validations HERE Hackathon of 2025.

Private presentation: https://uic365-my.sharepoint.com/:p:/g/personal/cwest23_uic_edu/EQUnWS7y6lFBo6q4lI4QhDoBul2Td-mYC9kCEUnlVkuYrw?e=4%3alZSzyN&at=9

## Features

- Detect and correct Geospatial inaccuracies in real-time
- Provide a friendly, interactive dashboard to view the corrections being made.
- Provide verbose logging for troubleshooting classifications.

## Tech Stack

### Backend
- Keras model from Teachable Machines
- Flask for the RESTful API
- Python for the data processing and classifying

### Frontend
- HTML
- JS
- CSS

## Prerequisites

- Python 3.13+ installed
- The close-source HERE dataset provided to compete in the hackathon

## Project Structure

```
├── LICENSE
├── README.md
├── app.py
├── config.py
├── coord.py
├── datasets
├── gm.py
├── helper.py
├── index.css
├── index.html
├── index.js
├── keras_model.h5
├── labels.txt
├── motorway-example.png
├── requirements.txt
├── road_attribution_correction.py
├── road_classifier.py
├── road_segment_corrector.py
├── run_task.py
└── sign_existence_corrector.py
```
## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
