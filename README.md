# The-Neural-Ear-to-Unearth-the-Composer-of-your-Music
The Neural Ear: Unearth the Composer of your Music | Create a deep-learning-based system that can classify the musical composer of the musical score. The project will only explore four well-known individuals in Western classical music - Bach, Beethoven, Chopin, and Mozart.

# Contents

1.  [Repository Name](#repository-name)
2.  [Title of the Project](#title-of-the-project)
3.  [Short Description and Objectives of the Project](#short-description-and-objectives-of-the-project)
4.  [Details about the Dataset](#details-about-the-dataset)
5.  [Goal of this Project](#goal-of-this-project)
6.  [Project Requirements](#project-requirements)
7.  [Usage Instructions in Local System and Kaggle](#usage-instructions-in-local-system-and-kaggle)
8.  [Detailed Conclusion with points](#detailed-conclusion-with-points)
9.  [Key Takeaways](#key-takeaways)
10. [Future Improvements](#future-improvements)
11. [Authors](#authors)
12. [References](#references)
----------------------------------------------

# Repository Name
The-Neural-Ear-to-Unearth-the-Composer-of-your-Music

----------------------------------------------

# Title of the Project
The Neural Ear: Unearth the Composer of your Music

----------------------------------------------

# Short Description and Objectives of the Project
- ***Description:*** The goal of this project is to create a deep-learning-based system that can classify the musical composer of the musical score. The project will only explore four well-known individuals in Western classical music - Bach, Beethoven, Chopin, and Mozart. 
  
- ***Objectives:*** The project will utilize the midi_classic_music dataset available from Kaggle that hosted 3929 midi files of classical musical scores with 175 different composers including Bach, Beethoven, Mozart, Brahms, Chopin, Tschaikovsky, Strauss, Stravinski, Prokofiev, Rachmaninov, Bernstein, Bartok, Handel, Ravel, Scriabin, etc. The challenge will be to build a good classification model with two state-of-the-art deep learning models:  Convolution Neural Networks (CNN) and Long Short-Term Memory Networks (LSTM) as these two models were chosen for their ability to learn spatial patterns and temporal patterns from music.

> The MIDI file is not an audio recording. Rather, it’s a recipe to an audio that explains what notes to play, when to play etc. It is a set of instructions such as pitch, note, tempo etc., that contain musical performance data used for electronic music production

----------------------------------------------

# Details about the Dataset

- **Name of the Dataset:** MIDI Classic Music
- **Description of the dataset:** Speech audio-only files (16bit, 48kHz .wav) from the RAVDESS. 

- As mentioned in the Dataset description in Kaggle, it contains 3929 midi files of classical works by 175 composers including Bach, Beethoven, Mozart, Brahms, Chopin, Tchaikovsky, Strauss, Stravinski, Prokofiev, Rachmaninov, Bernstein, Bartok, Handel, Ravel, Scriabin, and others.
	- Kaggle - https://www.kaggle.com/datasets/blanderbuss/midi-classic-music/data
	
- **Number of Variables:** Because the dataset is MIDI, and thus not a tabular CSV, there are not "columns" or "variables" to speak of. Rather, the information is encoded in the MIDI format, and from the MIDI files, the following features can be extracted:
    - Note pitch values
    - Note onset (start time) and offset (duration)
    - Tempo
    - Velocity (intensity of note)
    - Time signature & key signature (when available)
    - Chord progressions and patterns

- The class label (i.e. composer name) is typically inferred from the file name or folder structure of each MIDI file

- **Size of the Dataset:** The dataset approximately comprises of 122 MB of MIDI files.

----------------------------------------------

# Goal of this Project
The primary goal is to create a system that analyzes musical pieces and predicts the composer. There are several important steps to follow.
    - **Data Collection:** Gathering a diverse data set of musical pieces that are each associated with a composer.
    - **Feature Extraction:** Extracting meaningful features from the music, such as melody, rhythm, and harmony, that can differentiate composers.
    - **Model Construction:** Modeling the data with machine learning methods, possibly neural networks, to learn the patterns that will differentiate one composer from another.
    - **Evaluation:** Evaluating the model with appropriate metrics to validate its accuracy and reliability.

----------------------------------------------

# Project Requirements
- numpy
- pandas
- pretty-midi
- matplotlib
- seaborn
- music21
- tqdm
- scikit-learn
- tensorflow
- keras-tuner

----------------------------------------------

# Usage Instructions in Local System and Kaggle
- Clone using HTTPS
```commandline
git clone [https://github.com/kraviteja95usd/The-Neural-Ear-to-Unearth-the-Composer-of-your-Music.git](https://github.com/kraviteja95usd/The-Neural-Ear-to-Unearth-the-Composer-of-your-Music.git)
```
OR - 

- Clone using SSH
```commandline
git clone git@github.com:kraviteja95usd/The-Neural-Ear-to-Unearth-the-Composer-of-your-Music.git
```

OR -

- Clone using GitHub CLI
```commandline
gh repo clone kraviteja95usd/The-Neural-Ear-to-Unearth-the-Composer-of-your-Music
```
 
- Switch inside the Project Directory
```commandline
cd The-Neural-Ear-to-Unearth-the-Composer-of-your-Music
```

- Install Requirements in your local (if you run the notebook in your local)
```commandline
pip3 install -r requirements.txt
```

- Follow these guidelines to execute in Kaggle:
  - Import the [notebook](https://github.com/kraviteja95usd/The-Neural-Ear-to-Unearth-the-Composer-of-your-Music/blob/main/AAI-511-IN2_Final_Project_Files/AAI-511-IN2_Final%20Project%20Team-13_Music_Composer.ipynb) into your Kaggle account.
  - Once it is loaded, you will find an option to import the dataset. Look for the midi-classic-music and just import it. It loads to your `/kaggle/input` path.
  - Run all the cells of the notebook. You can view all the results
  - **Note-1:** While you execute the `Data Augmentation` part of `Data Pre-processing` section of the code, you will notice the zip files get generated. Download them.
  - **Note-2:** While you execute the `Feature Extraction` part of `Data Pre-processing` section of the code, you will notice a pickle file named `midi_files_feature_extractions.pkl` gets generated. Download it.
  - If you want to a view a sample of the same, you can find them here:
    - Augmented files: [Click Here](https://drive.google.com/drive/folders/1v80OhgkjscCW0PJxl6NOJhymUy2F9z3l?usp=sharing). 
      - You have to consider the following files.
        ```
        Mozart_augmented.zip
        Chopin_augmented.zip
        Bach_augmented.zip
        Beethoven_augmented.zip
        ```
    - Features Extracted File: [Click Here](https://drive.google.com/file/d/1LvJnRezoUb1xOabBVrngek4uC5_roRPc/view?usp=sharing).
      - You have to consider the `midi_files_feature_extractions.pkl` file.

----------------------------------------------

# Detailed Conclusion with points

**Summary Comparison Table of Best CNN and LSTM Models**

| Aspect                       | Best CNN Model (Hyperparameter Tuned)                | Best LSTM Model (Finetuned Bidirectional LSTM)          |
|------------------------------|------------------------------------------------------|----------------------------------------------------------|
| **Model Type**                       | CNN with tuned conv layers, dropout, LR optimized     | Bidirectional LSTM with dropout, LR scheduling         |
| **Total Parameters**                 | Moderate (~few hundred thousand, exact count varies)  | ~550,000                                              |
| **Training Epochs**                  | ~10-20                                               | 24 (with learning rates reduction)                    |
| **Test Accuracy**                    | **80.19%**                                          | 75.00%                                               |
| **Validation Accuracy**              | ~80%                                                | ~74%                                                 |
| **Macro F1-score**                   | 0.62                                               | 0.62                                                |
| **Weighted F1-score**                | 0.80                                               | 0.76                                                |
| **Dealing with Class Imbalance**     | Some (some dropout and tuning, no explicit weighting) | Used explicit methods (class weighting and dropout)   |
| **Performance on minority classes** | Moderate (decent F1 scores ~0.38 to 0.69),           | Same recall and F1 scores (~0.44 to 0.68) and better overall balance |
| **Overfitted behavior**              | Some overfitting as normal in base CNN, though controlled by tuning (dropout, batch size)  | Less overfitting overall, due to learning rate scheduling and dropout   |
| **Model Complexity**                 | Moderate (faster inference time in practice)         | Moderate to large, more computationally expensive   |
| **Best Use Case**                    | If spatial/local feature extraction is important      | If sequence/context modeling is important |

----------------------------------------------

# Key Takeaways

**Final Best Model Selection:**

| Model                         | Justification                                               |
|-------------------------------|-------------------------------------------------------------|
| **CNN Hyperparameter Tuned** | - Best overall test accuracy (80.19%).                         |
|                             | - Best weighted F1-score (0.80), indicating similar scores for all classes. |
|                             | - Good tuning on convolutional kernels and dropout for generalization. |
|                             | - Optimal model size, quicker training and inference.             |
| **Finetuned Bidirectional LSTM** | - Better suited for manipulating sequence data with imbalanced classes.          |
|                             | - Better minority class F1-scores, improved recall rate (to the class of interest mainly with classes 1, 2).           |
|                             | - Somewhat lower overall accuracy (75%) and would appear to have a better score on the metrics on minority classes. |
|                             | - Model is more complicated and goes through longer training.       |

----------------------------------------------

# Future Improvements

| Area                        | Suggestions                                                                                |
|----------------------------|-------------------------------------------------------------------------------------------|
| **Class Imbalance Handling**| - Utilize explicit class weighting or focal loss during CNN training to help minority class recall rates (e.g. possible to improve a minority classes recall rates from ± 55% to 70%).|  
|                            | - Apply data augmentation techniques to minority classes to improve diversity and augmented minority class input distributions.|
| **Model Architecture**      | - Experiment with hybrid CNN + LSTM architectures that train on 2 data modalities simultaneously to learn both spatial and sequential features.|
|                            | - Investigate transformer-based models to discover ways to improve modeling multi-modal context in time.| 
| **Regularization & Optimization** | - Utilize learning rate search, scheduling, most effective in CNN models and use early stopping more in both models.| 
|                            | - Use advanced regularization techniques (weight decay, batch norm, etc.) in both CNN and LSTM models.|  
| **Ensemble Methods**        | - Employ an ensemble method to help combine the predictions from the CNN and LSTM, i.e, use the strengths of both models to make predictions.|
| **Training with Augmented Data**| - Due to powerful resource constraints, we could not train with augmented data along with few more Deep Learning models. People whoever wish to consider this as a reference, think of an opportunity to experiment with diverse models, that too with the augmented data.|
| **Hyperparameter Search**   | - Fully automate hyperparameter search using techniques such as Bayesian optimization to improve inspection and coverage of hyperparameter search combinations.|
| **Explainability**          | - Apply model interpretability techniques to visually and qualitatively learn about the features of the model that differentiate classed as well as errors made based upon what part of the input the model focused on when arriving at its predictions.|
| **Deployment Considerations**| - Optimize for size and latency for on- or embedded deployment to allow for real-time responses.|

----------------------------------------------

# Authors

| Author            | Contact Details       |
|-------------------|-----------------------|
| Ravi Teja Kothuru | rkothuru@sandiego.edu |

----------------------------------------------

# References
- Bohdan Fedorak. (2019). midi_classic_music. Kaggle.com. https://www.kaggle.com/datasets/blanderbuss/midi-classic-music/data
- Ramsey, C., Huang, C., & Costa, D. (n.d.). The Classification of Musical Scores by Composer. https://cs230.stanford.edu/projects_fall_2018/reports/12441334.pdf
