# Description: Object detection using TensorFlow
import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model and labels
model_path = 'path/to/your/model.pb'  # Path to the frozen inference graph
label_path = 'path/to/your/labels.pbtxt'  # Path to the label map file
num_classes = 90  # Number of object classes

# Load the frozen TensorFlow model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.io.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load the label map
label_map = {}
with open(label_path, 'r') as f:
    for line in f:
        if 'id:' in line:
            id_val = int(line.strip().split(":")[-1])
        elif 'name:' in line:
            name_val = line.strip().split(":")[-1].strip().strip("'")
            label_map[id_val] = name_val

# Open a video capture object
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Define the input and output tensors
input_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
output_tensors = [
    detection_graph.get_tensor_by_name('detection_boxes:0'),
    detection_graph.get_tensor_by_name('detection_scores:0'),
    detection_graph.get_tensor_by_name('detection_classes:0'),
    detection_graph.get_tensor_by_name('num_detections:0')
]

# Start the object detection loop
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            # Read frame from the video capture
            ret, frame = cap.read()

            # Preprocess the frame
            image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Run the object detection
            output = sess.run(output_tensors, feed_dict={input_tensor: image_np_expanded})

            # Extract the detection results
            boxes = output[0][0]
            scores = output[1][0]
            classes = output[2][0].astype(np.int32)
            num_detections = int(output[3][0])

            # Draw bounding boxes on the frame
            for i in range(num_detections):
                if scores[i] > 0.5:  # Set a threshold for confidence score
                    ymin, xmin, ymax, xmax = boxes[i]
                    class_id = classes[i]
                    class_name = label_map[class_id]
                    score = scores[i]

                    height, width, _ = frame.shape
                    left = int(xmin * width)
                    top = int(ymin * height)
                    right = int(xmax * width)
                    bottom = int(ymax * height)

                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                    cv2.putText(frame, f"{class_name} ({score:.2f})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (36, 255, 12), 2)

            # Display the resulting frame
            cv2.imshow('Object Detection', frame)

            if cv2.waitKey(1) == ord('q'):
                break

# Release resources
cap.release()
cv2.destroyAllWindows()