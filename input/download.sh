#!/bin/sh

# ref: https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/data
echo 'download compe data...'
kaggle competitions download -c rsna-2023-abdominal-trauma-detection

# ref: https://www.kaggle.com/datasets/theoviel/rsna-abdominal-trauma-detection-png-pt1
echo 'download png...'
kaggle datasets download -d theoviel/rsna-abdominal-trauma-detection-png-pt1
kaggle datasets download -d theoviel/rsna-abdominal-trauma-detection-png-pt2
kaggle datasets download -d theoviel/rsna-2023-abdominal-trauma-detection-pngs-3-8
kaggle datasets download -d theoviel/rsna-abdominal-trauma-detection-png-pt4
kaggle datasets download -d theoviel/rsna-abdominal-trauma-detection-png-pt5
kaggle datasets download -d theoviel/rsna-abdominal-trauma-detection-png-pt6
kaggle datasets download -d theoviel/rsna-abdominal-trauma-detection-pngs-pt7
kaggle datasets download -d theoviel/rsna-2023-abdominal-trauma-detection-pngs-18
