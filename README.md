# Code-Lab-ML-Assignments

## Assignment 01/

### REQUIREMENTS :

-Read CSV file into a DataFrame using Pandas.

-Drop first column (First Name).

-Fill in missing values in (Salary, Bonus %) using the mean value of each column.

-Convert text values to numerical values in (Gender, Team).

-Create a dictionary for the mapping of the Team column (e.g. {'Marketing': 0, 'Others': 1, 'Finance': 2....}).

-Save the modified DataFrame to a CSV file.




## Assignment 02/

### DATASET:

-Breast cancer dataset where you use the features to predict whether the tumor is malignant (خبيث) or benign (رحيم).

-The dataset can be found here: https://drive.google.com/.../1ZW7WTKJ7j4vDnD6sqxO.../view...

### REQUIREMENTS:

-Read CSV file into a DataFrame using Pandas and do the necessary preprocessing steps.

-Split the data into inputs/features (x) and output/target ( y )

-Pick the appropriate linear model type for this kind of problem

-Train the model

-Evaluate model accuracy (should be above 90% or 0.90)




## Assignment 03/

### DATASET:

-We will be using the wine quality dataset, it's built into Scikit Learn so you can call it from the datasets module `datasets.load_wine()`.

-You will build a model that takes in the different features of wine and outputs a quality score from 0 to 2 (discrete values, not continuous)

-This dataset doesn't require any preprocessing so you can focus on applying cross validation and tuning the hyper parameters.

### REQUIREMENTS:

-Set random_state to 42 for every operation that accepts it.

-To split into training/testing set, use StratifiedShuffleSplit with split size of 10. Don't forget to set the random state to 42.

-Define your preferred model/algorithm for this task. Don't forget to set the random state to 42.

-Calculate the accuracy score for each fold, this dataset has multiple target classes so precision and recall aren't relevant here.

-Display the score for each fold and then the mean of these scores.

-Keep tuning the hyper-parameters until you reach an accuracy of 95% or more.




## Assignment 04/
-For this assignment, you will be using 2 different classification datasets and create a fully connected (Dense) neural network to predict the output based on the input features.

-The goal of this assignment is to get you used to using TensorFlow to build models. You don't need to worry about cross validation and hyper parameters tuning (except for the network size/depth).

### DATASETS:

-You will be using the following datasets from Scikit Learn:

-`datasets.load_iris`

-`datasets.load_wine`

### REQUIREMENTS:

-Set the random seed for TensorFlow using `tf.random.set_seed(42)` to ensure having reproducible results.

-Split your data using `model_selection.train_test_split` from Scikit Learn, use test size of 0.1 and make sure to stratify the targets and use a random_state of 42.

-One Hot encode your targets/outputs.

-Build and train a different models for each of the above datasets.

-Set the test set in the fit method using `validation_data=(x_test, y_test)`.

-Use accuracy metric in the compile function to display accuracy during training.

-Use Adam optimizer and CategoricalCrossentropy loss

-Make sure you're not overfitting/underfitting!




## Assignment 05/

You will use the Fashion MNIST dataset to build two deep learning models, one implemented as a Fully Connected Network (FCN/DNN), and one as a Convolutional Neural Network (CNN).

### DATASETS:

-The dataset is available at: `tf.keras.datasets.fashion_mnist.load_data()`

-and use `tf.data` to load the data into the fit method.

### REQUIREMENTS:

-Set the random seed for TensorFlow using `tf.random.set_seed(42)` to ensure having reproducible results.

-One Hot encode your targets/outputs.

-Set batch size to 32 using `tf.data` pipeline.

-Set the test set in the fit method using `validation_data=(x_test, y_test)`.

-Use accuracy metric in the compile function to display training/validation accuracy during training.

-Use the appropriate optimizer and loss function

-Achieve a training accuracy of at least 85% and validation accuracy of 80% using DNN, and ~100% training accuracy and above 95% validation accuracy in CNN.

-Plot the loss/validation loss value over the training epochs for both models.


## Assignment 06/
You'll be creating a model that takes in a sentence and predicts the appropriate emoji that describes the sentiment (similar to the IMDB example we did this week but with extra sentiments (6 instead of just 2)).
### DATASETS:
The dataset consists of two csv files, a training file with 16k rows and a testing file with 2k rows, each row has 3 columns, the sentence, the emotion as text (meant to provide description to the emoji and not to be used in training/testing) and the emoji symbol (e.g. 😄, 😡, 😍).
The dataset is available here: https://drive.google.com/.../1i7LmIH7sJHSARAMXgznBqXZ82tz...
### REQUIREMENTS:
-Set TensorFlow's random seed to 42

-Read data from CSV files and split it into inputs and targets (no need to do train_test_split as the data is already split).

-Tokenize and pad the text, use a vocabulary size of 10,000 and maximum sequence length of 64.

-Do the appropriate operations on the targets to prepare them for training.

-Define the sequential model, make sure to use LSTM and Embedding layers.

-Use Adam optimiser and the appropriate loss function and metrics to compile the model.

-Use ModelCheckpoint callback to save the model at the epoch with the best validation accuracy, the model file should have your name (e.g. for Sabri Monaf, the model's file name would be "sabri_m.h5). Make sure to download the model file as it will be a part of your assignment submission. (the documentation for ModelCheckpoint can be found here: https://www.tensorflow.org/.../callbacks/ModelCheckpoint).

-The training and validation accuracy should be at least 90% and 85% respectively.

-Create a cell in the end that takes in an input string (inputted using Colab Forms (see details here: https://colab.research.google.com/notebooks/forms.ipynb) and uses the trained model to predict (make sure to convert the vector output to the appropriate emoji and print only the emoji at the end of the cell).
