import os
import cv2
import time
from openvino.inference_engine import IECore
import ngraph as ng
#%matplotlib inline
from matplotlib import pyplot as plt
print('Imported Python modules successfully.')

# model IR files
model_xml = "model/public/mobilenet-ssd/FP32/mobilenet-ssd.xml"
model_bin = os.path.splitext(model_xml)[0] + ".bin" # create IR .bin filename from path to IR .xml file

# input image file
input_path = "car.bmp"

# CPU extension library to use
cpu_extension_path = os.path.expanduser("~")+"/inference_engine_samples/intel64/Release/lib/libcpu_extension.so"

# device to use
device = "CPU"

# output labels 
labels_path = "labels.txt"

# minimum probability threshold to detect an object
prob_threshold = 0.5

print("Configuration parameters settings:"
     "\n\tmodel_xml=", model_xml,
      "\n\tmodel_bin=", model_bin,
      "\n\tinput_path=", input_path,
      "\n\tdevice=", device, 
      "\n\tlabels_path=", labels_path, 
      "\n\tprob_threshold=", prob_threshold)

ie = IECore()
print("An inference engine object has been created")

# load network from IR files
net = ie.read_network(model=model_xml, weights=model_bin)
print("Loaded model IR files [",model_bin,"] and [", model_xml, "]\n")

# check to make sure that the plugin has support for all layers in the loaded model
supported_layers = ie.query_network(net, "CPU") 
ng_function = ng.function_from_cnn(net)
not_supported_layers = \
[node.get_friendly_name() for node in ng_function.get_ordered_ops() if node.get_friendly_name() not in supported_layers]
if len(not_supported_layers) != 0:
    log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(device, ', '.join(not_supported_layers)))
    log.error("Please try to specify cpu extensions library path in sample's command line parameters using"
                      "--cpu_extension command line argument")
    sys.exit(1)

# check to make sue that the model's input and output are what is expected
assert len(net.input_info.keys()) == 1, \
    "ERROR: This sample supports only single input topologies"
assert len(net.outputs) == 1, \
    "ERROR: This sample supports only single output topologies"
print("SUCCESS: Model IR files have been loaded and verified")

# load the model into the Inference Engine for our device
exec_net = ie.load_network(network=net, num_requests=2, device_name=device)

# store name of input and output blobs
input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))

# read the input's dimensions: n=batch size, c=number of channels, h=height, w=width
n, c, h, w = net.input_info[input_blob].input_data.shape
print("Loaded model into Inference Engine for device:", device, 
      "\nModel input dimensions: n=",n,", c=",c,", h=",h,", w=",w)

labels_map = None
# if labels points to a label mapping file, then load the file into labels_map
print(labels_path)
if os.path.isfile(labels_path):
    with open(labels_path, 'r') as f:
        labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    print("Loaded label mapping file [",labels_path,"]")
else:
    print("No label mapping file has been loaded, only numbers will be used",
          " for detected object labels")

# define function to load an input image
def loadInputImage(input_path, verbose = True):
    # globals to store input width and height
    global input_w, input_h
    
    # use OpenCV to load the input image
    cap = cv2.VideoCapture(input_path) 
    
    # store input width and height
    input_w = cap.get(3)
    input_h = cap.get(4)
    if verbose: print("Loaded input image [",input_path,"], resolution=", input_w, "w x ",input_h,"h")

    # load the input image
    ret, image = cap.read()
    del cap
    return image

# define function for resizing input image
def resizeInputImage(image, verbose = True):
    # resize image dimensions form image to model's input w x h
    in_frame = cv2.resize(image, (w, h))
    # Change data layout from HWC to CHW
    in_frame = in_frame.transpose((2, 0, 1))  
    # reshape to input dimensions
    in_frame = in_frame.reshape((n, c, h, w))
    if verbose: print("Resized input image from {} to {}".format(image.shape[:-1], (h, w)))
    return in_frame

# load image
image = loadInputImage(input_path)

# resize the input image
in_frame = resizeInputImage(image)

# display input image
print("Input image:")
plt.axis("off")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

inf_start = time.time()
res = exec_net.infer(inputs={input_blob: in_frame})
inf_time = time.time() - inf_start
print("Inference complete, run time: {:3f} ms".format(inf_time * 1000))
# create function to process inference results
def processResults(result):
    # get output results
    res = result[output_blob]
    
    # loop through all possible results
    for obj in res[0][0]:
        # If probability is more than specified threshold, draw and label box 
        if obj[2] > prob_threshold:
            # get coordinates of box containing detected object
            xmin = int(obj[3] * input_w)
            ymin = int(obj[4] * input_h)
            xmax = int(obj[5] * input_w)
            ymax = int(obj[6] * input_h)
            
            # get type of object detected
            class_id = int(obj[1])
            
            # Draw box and label for detected object
            color = (min(class_id * 12.5, 255), 255, 255)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 4)
            det_label = labels_map[class_id] if labels_map else str(class_id)
            cv2.putText(image, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                        cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

processResults(res)
print("Processed inference output results.")

# convert colors BGR -> RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# disable axis display, then display image
plt.axis("off")
plt.imshow(image)
