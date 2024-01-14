#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow numpy matplotlib


# In[2]:


# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[3]:


# Define paths to your training and validation datasets
train_data_dir = "C:/Deep Learning/cotton/TRAIN"
validation_data_dir = "C:/Deep Learning/cotton/TEST"


# In[4]:


# Set parameters
img_width, img_height = 224, 224
batch_size = 30
epochs = 10


# In[5]:


# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


# In[6]:


# Rescale validation data
validation_datagen = ImageDataGenerator(rescale=1./255)


# In[7]:


# Load training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)


# In[8]:


# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)


# In[9]:


# Define GoogLeNet (Inception) model
base_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(img_width, img_height, 3),
    pooling=None
)


# In[10]:


# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False


# In[11]:


# Define the number of classes in your classification problem
num_classes = len(train_generator.class_indices)


# In[12]:


# Create a custom model on top of the base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])


# In[13]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[14]:


# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)



# In[15]:


# Evaluate the model
evaluation = model.evaluate(validation_generator)
print(f"Validation Accuracy: {evaluation[1] * 100:.2f}%")


# In[16]:


# Save the model
model.save('cotton_disease_model.h5')


# In[60]:


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import numpy as np


# In[61]:


# Load the pre-trained GoogLeNet (Inception) model
model = InceptionV3(weights='imagenet')


# In[84]:


# Load and preprocess an image for prediction
img_path = "C:/Deep Learning/cotton/TEST/test/5586201.jpg"
img = image.load_img(img_path, target_size=(299, 299))  # GoogLeNet input size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)


# In[85]:


# Make predictions
predictions = model.predict(img_array)


# In[86]:


model = tf.keras.models.load_model('cotton_disease_model.h5')


# In[87]:


# Function to preprocess an image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to [0, 1]
    return img_array


# In[88]:


import matplotlib.pyplot as plt


# In[89]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('cotton_disease_model.h5')

# Function to preprocess an image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to [0, 1]
    return img_array

# Function to make predictions and display results
def predict_and_display(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)

    # Make predictions
    predictions = model.predict(img)
    class_labels = ['Diseased', 'Healthy']

    # Display the original image
    img = image.load_img(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Display prediction results
    print("Predictions:")
    for i in range(len(class_labels)):
        print(f"{class_labels[i]}: {predictions[0][i]:.4f}")

    predicted_label = class_labels[np.argmax(predictions)]
    print(f"Predicted Label: {predicted_label}")

# Example usage
image_path ="C:/Deep Learning/cotton/TEST/test/5586201.jpg"
predict_and_display(image_path)


# In[106]:


history = history

# Accessing the training and validation accuracy
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Number of epochs
EPOCHS = len(train_acc)

# Plotting the accuracy
plt.figure(figsize=(14, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), train_acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.show()


# In[90]:


from sklearn.metrics import confusion_matrix, classification_report


# In[91]:


# Define paths to your test dataset
test_data_dir = "C:/Deep Learning/cotton/TEST"


# In[93]:


# Set parameters
img_width, img_height = 299, 299  # GoogLeNet input size
batch_size = 30


# In[94]:


# Data preprocessing for testing
test_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)


# In[95]:


# Load testing data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=False,  # Important: Keep the order of predictions consistent with ground truth
    class_mode='categorical'
)


# In[96]:


# Make predictions on the test set
predictions = model.predict(test_generator)


# In[97]:


# Decode predictions and ground truth labels
predicted_labels = np.argmax(predictions, axis=1)
true_labels = test_generator.classes


# In[98]:


# Compute confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)


# In[99]:


# Plot confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')


# In[105]:


# Print classification report
print("Classification Report:")
print(classification_report(true_labels, predicted_labels))

