import cv2
import numpy as np

import tensorflow as tf

import imutils
from cv_utils import *
from db_CSV import *



class NeuralNetwork:
    def __init__(self):
        self.model_file = "./model/binary_128_0.50_ver3.pb"
        self.label_file = "./model/binary_128_0.50_labels_ver2.txt"
        self.label = self.load_label(self.label_file)
        self.graph = self.load_graph(self.model_file)
        self.sess = tf.Session(graph=self.graph)

    def load_graph(self, modelFile):
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with open(modelFile, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    def load_label(self, labelFile):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(labelFile).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def convert_tensor(self, image, imageSizeOuput):
        """
    takes an image and tranform it in tensor
    """
        image = cv2.resize(image, dsize=(imageSizeOuput, imageSizeOuput), interpolation=cv2.INTER_CUBIC)
        np_image_data = np.asarray(image)
        np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        np_final = np.expand_dims(np_image_data, axis=0)
        return np_final

    def label_image(self, tensor):

        input_name = "import/input"
        output_name = "import/final_result"

        input_operation = self.graph.get_operation_by_name(input_name)
        output_operation = self.graph.get_operation_by_name(output_name)

        results = self.sess.run(output_operation.outputs[0],
                                {input_operation.outputs[0]: tensor})
        results = np.squeeze(results)
        labels = self.label
        top = results.argsort()[-1:][::-1]
        return labels[top[0]]

    def label_image_list(self, listImages, imageSizeOuput):
        plate = ""
        for img in listImages:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            plate = plate + self.label_image(self.convert_tensor(img, imageSizeOuput))
        return plate, len(plate)


# out = cv2.VideoWriter('Numberplate_out.mp4',cv2.VideoWriter_fourcc(*'MP4V'),20, (720,420))
frame_array = []
if __name__ == "__main__":
    findPlate = PlateFinder()

    # Initialize the Neural Network
    model = NeuralNetwork()

    cap = cv2.VideoCapture('vid1.MP4')
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)  
    cv2.resizeWindow("output", 720, 420) 
    # frameNo=0    
    # detectionframeNo=-11
    while (cap.isOpened()):
        ret, img = cap.read()
        # frameNo=frameNo+1
        if ret == True:
            
            # if detectionframeNo < frameNo - 10:
            possible_plates,cord = findPlate.find_possible_plates(img)
            if possible_plates is not None:
                for i, p in enumerate(possible_plates):
                    x,y,w,h=cord[i]
                    print('cord[i] ', cord[i])                    
                    chars_on_plate = findPlate.char_on_plate[i]
                    recognized_plate, _ = model.label_image_list(chars_on_plate, imageSizeOuput=128)
                    
                    print(recognized_plate)
                    Name, Contact, Address,Status=db_get_val(recognized_plate)
                    Text=Name + ' , '+ Contact + ' , ' +Status
                    if Name is not '0':
                        cv2.rectangle(img, pt1=(50,50), pt2=(1900,120), color=(0,0,0), thickness= -1)
                        cv2.putText(img, Text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 3)
                        # cv2.putText(img, Status, (x+200, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
                    
                    cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 3)
                    cv2.putText(img, recognized_plate, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)

                    

                    # detectionframeNo=frameNo
            cv2.imshow('output',img)          
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        


        
        else:
            break
            
        


cap.release()
cv2.destroyAllWindows()
