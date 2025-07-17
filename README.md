# DL_algorithms

## Basic_Deep_Learning_Ann.ipynb

### Basic Deep Learning with Artificial Neural Network (ANN)

This Jupyter Notebook provides a basic example of implementing an Artificial Neural Network (ANN) for a binary classification task using a bank marketing dataset.

### Overview

The notebook demonstrates the following key steps in building and training an ANN:

*   **Library Imports**: Importing necessary libraries for data manipulation, machine learning preprocessing, and TensorFlow/Keras for building the neural network.
*   **Data Loading**: Loading the `bank.csv` dataset into a Pandas DataFrame.
*   **Data Exploration**: Briefly examining the dataset and its columns, including a description of the features.
*   **Data Preprocessing**:
    *   Dropping the 'duration' column (as indicated in the notebook's comments, though the reason isn't explicitly stated, it's a common practice to remove features that might lead to data leakage or are not relevant for prediction in a real-world scenario).
    *   Checking the distribution of the target variable ('deposit').
    *   Encoding the target variable ('deposit') from 'yes'/'no' to 1/0.
    *   Separating features (X) and target (y).
    *   One-hot encoding categorical features using `pd.get_dummies`.
    *   Feature scaling numerical features using `StandardScaler`.
*   **Data Splitting**: Dividing the dataset into training and testing sets.
*   **Model Building**: Constructing a sequential ANN model with dense layers.
*   **Model Compilation**: Defining the loss function, optimizer, and metrics for the model.
*   **Model Training**: Training the ANN model on the prepared data.
*   **Visualization**: Plotting the training and validation accuracy and loss over epochs to assess model performance and identify overfitting/underfitting.

### Dataset Description (`bank.csv`)

The dataset contains information about bank customers and their interaction with a marketing campaign. Key features include:

*   **Age**: Age of the customer.
*   **Job**: Occupation of the customer.
*   **Marital Status**: Marital status of the customer.
*   **Education**: Education level of the customer.
*   **Default**: Whether the customer has credit in default.
*   **Balance**: Balance of the customer's account.
*   **Housing Loan**: Whether the customer has a housing loan.
*   **Contact Communication Type**: Method used to contact the customer (e.g., telephone, cellular).
*   **Day**: Day of the month when the last contact was made.
*   **Duration**: Duration (in seconds) of the last contact during a campaign (this column is dropped in the notebook).
*   **Campaign Contacts Count**: Number of contacts performed during the current campaign.
*   **pdays**: Number of days passed since the customer was last contacted from a previous campaign.
*   **poutcome**: Outcome from the previous marketing campaign.
*   **Deposit (Target)**: Whether the customer subscribed to a term deposit ('yes' or 'no').

### Dependencies

*   `numpy`
*   `pandas`
*   `sklearn` (for `train_test_split`, `StandardScaler`)
*   `tensorflow`
*   `matplotlib`
*   `seaborn`

You can install these dependencies using pip:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

### How to Run

1.  **Download the notebook and the `bank.csv` dataset.** Ensure `bank.csv` is in the same directory as the notebook.
2.  **Ensure you have the required dependencies installed.**
3.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook Basic_Deep_Learning_Ann.ipynb
    ```
4.  **Run all cells** in the notebook.

### Model Architecture

The ANN model is a simple sequential model with:

*   An input `Dense` layer with 32 units and ReLU activation. The `input_dim` is set dynamically based on the number of features after one-hot encoding.
*   A hidden `Dense` layer with 16 units and ReLU activation.
*   An output `Dense` layer with 1 unit and `sigmoid` activation for binary classification.

### Training Details

*   **Loss Function**: `binary_crossentropy` (suitable for binary classification).
*   **Optimizer**: `SGD` (Stochastic Gradient Descent).
*   **Metrics**: `accuracy`.
*   **Epochs**: 150.
*   **Batch Size**: 32.
*   **Validation Split**: 10% of the training data is used for validation.

### Training Results

The training output shows the accuracy and loss for both training and validation sets across 150 epochs. The plots generated at the end of the notebook visualize these trends, which can help in understanding the model's learning process and potential issues like overfitting (where training accuracy continues to improve but validation accuracy plateaus or decreases). Based on the provided output, the validation accuracy hovers around 65-69%, while training accuracy reaches around 83%, suggesting some degree of overfitting or that the model might benefit from further tuning or a different architecture.

---

## CNN_MNIST_Digits.ipynb

### Convolutional Neural Network for MNIST Digit Recognition

This Jupyter Notebook demonstrates the implementation of a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset.

### Overview

The notebook covers the following steps:

*   **Data Loading**: Loading the MNIST dataset using TensorFlow/Keras.
*   **Data Preprocessing**: Normalizing and reshaping the image data to be suitable for a CNN.
*   **Model Building**: Constructing a sequential CNN model with convolutional, pooling, flattening, and dense layers.
*   **Model Compilation**: Configuring the model with an optimizer, loss function, and metrics.
*   **Model Training**: Training the CNN on the training data and evaluating its performance on the test set.
*   **Prediction on Custom Image**: Demonstrating how to load a custom image, preprocess it, and use the trained model to predict the digit.

### Dependencies

*   `tensorflow`
*   `matplotlib`
*   `Pillow` (PIL)
*   `numpy`
*   `pandas` (though `pandas` is imported, it's not explicitly used in the core CNN digit recognition logic, but might be for general data handling if the notebook were expanded).

You can install these dependencies using pip:

```bash
pip install tensorflow matplotlib Pillow numpy pandas
```

### How to Run

1.  **Clone the repository (if applicable) or download the notebook.**
2.  **Ensure you have the required dependencies installed.**
3.  **Place a custom image named `four.jpg` (or any other digit image) in the same directory as the notebook if you wish to test the custom image prediction.** The example in the notebook uses `four.jpg`.
4.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook CNN_MNIST_Digits.ipynb
    ```
5.  **Run all cells** in the notebook.

### Model Architecture

The CNN model consists of:

*   Two `Conv2D` layers with ReLU activation and `MaxPooling2D` layers for feature extraction.
*   A `Flatten` layer to convert the 2D feature maps into a 1D vector.
*   Two `Dense` (fully connected) layers with ReLU activation.
*   An output `Dense` layer with 10 units (for 10 digits) and `softmax` activation for multi-class classification.

### Training Results

The model is trained for 5 epochs. The output from the training shows high accuracy on both training and validation data:

```
Epoch 1/5
... - accuracy: 0.8895 - loss: 0.3619 - val_accuracy: 0.9844 - val_loss: 0.0469
Epoch 2/5
... - accuracy: 0.9851 - loss: 0.0492 - val_accuracy: 0.9879 - val_loss: 0.0350
Epoch 3/5
... - accuracy: 0.9901 - loss: 0.0316 - val_accuracy: 0.9871 - val_loss: 0.0360
Epoch 4/5
... - accuracy: 0.9924 - loss: 0.0228 - val_accuracy: 0.9890 - val_loss: 0.0349
Epoch 5/5
... - accuracy: 0.9943 - loss: 0.0175 - val_accuracy: 0.9873 - val_loss: 0.0399
Test accuracy :  0.9873
```

The model achieves a test accuracy of approximately 98.73%.

---
