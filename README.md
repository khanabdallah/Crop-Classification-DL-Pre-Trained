# Crop Classification using Deep Learning

This project develops a crop classification model using deep learning and a pre-trained neural network. The goal is to accurately identify different crop types based on input images. The model uses transfer learning by leveraging the MobileNetV2 architecture, which has been pre-trained on a large dataset of images. This approach significantly speeds up training and improves accuracy by using a network that already understands visual patterns. The project also incorporates data augmentation and fine-tuning techniques to further optimize performance.

---

### Technologies Used

* **Python:** The core programming language for the entire project.

* **TensorFlow & Keras:** The deep learning framework used to build, train, and fine-tune the neural network.

* **MobileNetV2:** A pre-trained convolutional neural network (CNN) architecture used for transfer learning.

* **Matplotlib:** A plotting library used to visualize the model's training history, including accuracy and loss.

* **ImageDataGenerator:** A Keras utility used for real-time data augmentation and preprocessing.

---

### Key Features and Methodology

* **Transfer Learning:** The model utilizes the pre-trained MobileNetV2 base, allowing it to learn new features without training from scratch. This approach is highly efficient and effective.

* **Data Augmentation:** The `ImageDataGenerator` class is used to create a larger, more diverse training dataset by applying random transformations like rotation, zooming, and flipping. This helps prevent overfitting and improves the model's ability to generalize to new images.

* **Model Architecture:** A custom classifier is built on top of the MobileNetV2 base. It includes a `GlobalAveragePooling2D` layer to reduce dimensionality and a `Dropout` layer to prevent overfitting, followed by dense layers for final classification.

* **Fine-Tuning:** After initial training, the top layers of the pre-trained model are unfrozen and trained with a lower learning rate. This fine-tuning process allows the model to adjust the pre-trained weights to be more specific to the crop classification task, leading to higher accuracy.

* **Callbacks:** A `ReduceLROnPlateau` callback is used to automatically reduce the learning rate if the validation loss plateaus, helping the model converge more effectively.

---

### How to Run the Code

1.  **Prerequisites:** Ensure you have Python installed on your system.

2.  **Install Libraries:** Install the necessary Python libraries by running the following commands in your terminal:
    ```
    pip install tensorflow
    pip install matplotlib
    pip install pandas
    pip install numpy
    ```

3.  **Download the Dataset:** This project requires a dataset of crop images for training and validation. The dataset is not included in this repository due to its large size. You can download the dataset from this link:

    https://drive.google.com/drive/folders/16blYHZ2feoIjo-oSik7dDsZrhmHC-leZ?usp=drive_link

    Once downloaded, please place the data in the `archive/` folder, ensuring the directory structure matches the one specified in the code.

4.  **Run the Script:** Navigate to the directory containing the Python script and execute it:
    ```
    python your_script_name.py
    ```
    The script will start training the model and display the accuracy and loss plots after training is complete.
