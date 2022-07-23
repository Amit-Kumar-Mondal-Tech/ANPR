import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import numpy as np
import pytesseract
import cv2
import re



physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except Exception as e:
    print(e)



class ModelLoader:
    @classmethod
    def getModel(cls):
        configs = config_util.get_configs_from_pipeline_file(r"model/pipeline.config")
        detection_model = model_builder.build(
            model_config=configs["model"], is_training=False
        )

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(r"model/ckpt-56").expect_partial()
        category_index = label_map_util.create_category_index_from_labelmap(
            r"model/label_map.pbtxt"
        )
        return detection_model, category_index


class Detector:
    detection_model, category_index = ModelLoader.getModel()

    def processDetections(self, detections, threshold):
        num_detections = int(detections.pop("num_detections"))
        detections = {
            key: value[0, :num_detections].numpy() for key, value in detections.items()
        }
        detections["num_detections"] = num_detections

        # detection_classes should be ints.
        detections["detection_classes"] = detections["detection_classes"].astype(
            np.int64
        )

        scores = list(filter(lambda x: x > threshold, detections["detection_scores"]))
        boxes = detections["detection_boxes"][: len(scores)]
        classes = detections["detection_classes"][: len(scores)]
        return boxes, classes

    def labeling(self, image, threshold=0.75, max_boxes=5):
        detection_model = Detector.detection_model

        @tf.function
        def detect_fn(image):
            image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)
            return detections

        image_np = np.array(image)
        height, width = image_np.shape[:2]
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tf.float32
        )
        detections = detect_fn(input_tensor)

        boxes, classes = self.processDetections(detections, threshold)
        boxes = boxes * np.array([height, width, height, width])
        boxes = boxes.astype("uint32")
        # boxes = [box1, box2, box3, ,,,,,]
        # box = [w, x, y, z]
        # pt1 = box[1], box[0] upper left (x, w)
        # pt2 = box[3], box[2] lower right (z, y)
        for box in boxes:
        	continue

        return image
    
    def recognize_plate(self, image, box):
        xmin, ymin, xmax, ymax = box
        cropedImage = image[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]

        # preprocesing
        grayImage = cv2.cv2.cvtColor(cropedImage, cv2.COLOR_BGR2GRAY)
        grayImage = cv2.resize(grayImage, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
        blurImage = cv2.GaussianBlur(grayImage, (5,5), 0)
        ret, thresh = cv2.threshold(grayImage, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        dilation = cv2.dilate(thresh, rect_kern, iterations = 1)

        try:
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except Exception as e:
            ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        image2 = grayImage.copy()
        plate_num = ""
        
        for cnt in sorted_contours:
            x,y,w,h = cv2.boundingRect(cnt)

            height, width = image2.shape
            if height / h > 6:
            	continue

            ratio = h / w

            if ratio < 1.5:
            	continue

            if width / w > 15:
            	continue

            area = h * w
            if area < 100: 
            	continue

            rect = cv2.rectangle(image2, (x,y), (x+w, y+h), (0,255,0),2)
            roi = thresh[y-5:y+h+5, x-5:x+w+5]
            roi = cv2.bitwise_not(roi)
            roi = cv2.medianBlur(roi, 5)
            try:
                text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
                clean_text = re.sub(r'[\W_]+', '', text)
                plate_num += clean_text
            except: 
                text = None

        if plate_num != None:
            print("License Plate #: ", plate_num)

        return plate_num
        
