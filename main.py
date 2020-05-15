from gradcam import GradCAM
import tensorflow as tf
import numpy as np
import imutils
import cv2

image_path = 'data/train/COVID-19/1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-000-fig1a.png'
model = tf.keras.applications.ResNet50(weights='imagenet')

original = cv2.imread(image_path)
resized = cv2.resize(original, (224, 224))

image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
image = tf.keras.preprocessing.image.img_to_array(image)
image = np.expand_dims(image, axis=0)
image = tf.keras.applications.imagenet_utils.preprocess_input(image)

predictions = model.predict(image)
class_index = np.argmax(predictions[0])

decoded = tf.keras.applications.imagenet_utils.decode_predictions(predictions)
imagenetID, label, prob = decoded[0][0]
label = "{}: {:.2f}%".format(label, prob * 100)
print("[INFO] {}".format(label))

cam = GradCAM(model, class_index)
heatmap = cam.generate_heatmap(image)

heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
heatmap, output = cam.overlay_heatmap(heatmap, original, alpha=0.5)

cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
	0.8, (255, 255, 255), 2)
# display the original image and resulting heatmap and output image
# to our screen
output = np.vstack([original, heatmap, output])
output = imutils.resize(output, height=700)
cv2.imshow("Output", output)
cv2.waitKey(0)
