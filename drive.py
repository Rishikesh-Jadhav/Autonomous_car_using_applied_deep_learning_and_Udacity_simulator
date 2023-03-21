# import socketio
# import eventlet
# from flask import Flask
# from keras.models import load_model
# import base64
# from io import BytesIO
# from PIL import Image #Image.open()
# import numpy as np
# import cv2

# # web sockets are used to form real time connection btw cliets andd server
# #continous updating
# sio = socketio.Server()

# app = Flask('__name__') #'__main__'- when executed

# speed_limit =10

# #using the image preproces to feed the live data in our model
# def img_preprocess(img):
#   img = img[60:135,:,:]
#   img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV) # yuv 3 channels - y(luminosty/brightness),u,v(chromium components which add colors to the image)
#   #Gaussian blur for smoothening and reducing noise kernel convolution
#   img = cv2.GaussianBlur(img, (3,3), 0)
#   # resize to increase size for faster computation
#   img = cv2.resize(img, (200,66)) #(image,size of image)
#   #normalize(no visual immpact)
#   img = img/255
#   return img




# #simulator will start of with the following values inn send control facing forward and send data to us about the environment
# #This data willl give us the current image where the car is in the track.
# #based on this image model will extract the features of the image and predict steering angles 

# #special vent handler
# @sio.on('telemetry')
# def telemetry(sid,data):#(section id, data)
#     speed = float(data['speed'])
#     image = Image.open(BytesIO(base64.b64decode(data['image']))) # base 64 encoded so we need to decode it
#     #before we can decode and identify the given image file with image.open we need to take one more step
#     #Buffer module which can mimic our data as a normal file bytesIO

#     image=np.asarray(image)
#     image = img_preprocess(image) # model exoects 4d arrays wher our image is 3d
#     image = np.array([image]) #to add another dimention
#     throttle = 1.0-speed/speed_limit
#     steering_angle  = float(model.predict(image))
#     print('{} {} {}'.format(steering_angle,throttle,speed))
#     send_control(steering_angle,1.0)



# #register event handler
# @sio.on('connect') #'message','disconnect'
# def connect(sid, environ):
#     print('connected')
#     #initial steering and throttle values static, once our model recieves those it will process data and use send control with steering angles 
#     send_control(0, 0)

# #simulator emit the sendo control to the simulator
# def send_control(steering_angle, throttle):
#     sio.emit('steer', data = { 
#         'steering_angle': steering_angle.__str__(),
#         'throttle': throttle.__str__() # pass as a string

#     }) # send data in key value pairs

# # eg:-
# # #router decorator
# # @app.route('/home')
# # def greeting():
# #     return 'WELCOME!'
# # if we specify the url on local host 3000 port then it will run the function



# if __name__ == '__main__':
#     model = load_model('model.h5')
# #    app.run(port=3000)
#     # combine socket io server with the flask
#     app = socketio.Middleware(sio,app) # server sio with flask app
#     #web servr gateway interfacev -wsgi to have theserver send any requests made by client to web app

#     eventlet.wsgi.server(eventlet.listen(('',4567)),app)#listen opens listening sockets((ip,port), app to which the requests are gooing to be sent)('blank space is for any ip')



import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
from tensorflow.keras.models import  model_from_json

sio = socketio.Server()
 
app = Flask(__name__) #'__main__'
speed_limit = 10
def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
 
 
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)
 
 
 
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)
 
def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })
 
 
if __name__ == '__main__':
    json_file = open('C:/Users/rishi/OneDrive/Desktop/behavioral_cloninig/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("C:/Users/rishi/OneDrive/Desktop/behavioral_cloninig/model.h5")

    #model = load_model("C:/Users/rishi/OneDrive/Desktop/behavioral_cloninig/model.h5")
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)