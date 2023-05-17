import cv2
import tensorflow as tf
import numpy as np

def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

def result_image(conf, model, image):
    # Resize the image to a standard size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_to_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.uint8)
    output = model(img_to_tensor)
    score = output['detection_scores'][0]
    confidence_indexs = [i for i in range(len(score)) if score[i] > conf]
    boxes = output['detection_boxes'][0]
    confidence_boxes = [boxes[i].numpy().tolist() for i in confidence_indexs]
    dimensions = image.shape
    height = image.shape[0]
    width = image.shape[1]
    for box in confidence_boxes:
        start_point = (int(box[1] * height), int(box[0] * width))
        end_point = (int(box[3] * height), int(box[2] * width))
        color = (255,0,0)
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        image = cv2.putText(image, 'Crack', (int(box[1] * height), int(box[0] * width) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return image