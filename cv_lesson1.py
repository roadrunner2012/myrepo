import cv2,os

filename = 'video.avi' # Filename for recording the video
frames_per_seconds = 24.0
myres = '720p' #resolution



#change resulution

def make_1080p():
    cap.set(3,1920)
    cap.set(4,1080)

def change_res(cap,width,height):
    cap.set(3,width)
    cap.set(4,height)

#Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "480p":(640,480),
    "720p":(1280,720),
    "1080p":(1920,1080),
    "4k":(3840,2160)
}

def get_dims(cap, res='1080p'):
    width,height = STD_DIMENSIONS['480p']
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    change_res(cap,width,height)
    return width,height

# Video Encoding, might require additional installs
# Types of codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi' : cv2.VideoWriter_fourcc(*'XVID'),
##    'mp4' : cv2.VideoWriter_fourcc(*'H264'),
    'mp4' : cv2.VideoWriter_fourcc(*'XVID'),
    }

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

face_cascade=cv2.CascadeClassifier('/home/me/src/cascades/data/haarcascade_frontalface/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)
dims = get_dims(cap,res=myres)
video_type_cv2 = get_video_type(filename)
out = cv2.VideoWriter(filename,video_type_cv2,frames_per_seconds,dims)


#rescale_frame
def rescale_frame(frame, percent=75):
    scale_percent = 75
    width = int(frame.shape[1] * scale_percent /100)
    height = int(frame.shape[0] * scale_percent /100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

#make_1080p()


while True:
    #capturing frame-by-frame
    ret,frame = cap.read()
    #start recording
    out.write(frame)
    #scale it
    frame = rescale_frame(frame, percent=40)
    #creates special color set
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        img_item="my-image.png"
        cv2.imwrite(img_item,roi_gray)

        color=(255,0,0) #BGR
        stroke=2 #Strichstaerke
        end_cord_x=x+w
        end_cord_y=y+h
        cv2.rectangle(frame,(x,y), (end_cord_x,end_cord_y),color,stroke)
        




    #display the resulting frame
##    cv2.imshow('myframe',frame)
    cv2.imshow('myframe1',gray)
    #waits for keystorke
    if cv2.waitKey(20) & 0xFF == ord('q'): 
        break

# cleanup, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
