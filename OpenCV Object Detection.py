
# coding: utf-8

# In[38]:


import cv2 
import numpy as np
import matplotlib.pyplot as plt


# In[31]:


#1
def main():
    print(cv2.__version__)


# In[4]:


#2
#To acquire Images Use Web Cam or Pi camera using Rasperry 
def main():
    imgpath="E:\\clg imp\\DTU PROJECT\\Open CV\\standard_test_images\\lena_color_256.tif"
    
    img=cv2.imread(imgpath)
    cv2.namedWindow('Lena',cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Lena',img)
    cv2.waitKey(0)#To bind cv2 with keyboard event it will destroy all the windows if pressed
    cv2.destroyAllWindows()


# In[11]:


#3 TO create  and delete a single window
def main():
    imgpath="E:\\clg imp\\DTU PROJECT\\Open CV\\standard_test_images\\lena_color_256.tif"
    
    img=cv2.imread(imgpath)
    cv2.namedWindow('Lena',cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Lena',img)
    cv2.waitKey(0)#To bind cv2 with keyboard event it will destroy all the windows if pressed
    cv2.destroyWindow('Lena')


# In[28]:


#Modes of pic #1 is default if nothing uses then 1 wil be by default #0 is grayscale #-1 read img as it is 
#and store alphatransparency in img
#Read and write images from/to disk
def main():
    imgpath="E:\\clg imp\\DTU PROJECT\\Open CV\\standard_test_images\\lena_color_256.tif"
    
    #img=cv2.imread(imgpath,1)
    img=cv2.imread(imgpath,0)
    
    outpath="E:\\clg imp\\DTU PROJECT\\Open CV\\output\\lena_color_256.jpg"
    
    #cv2.namedWindow('Lena',cv2.WINDOW_AUTOSIZE)
    cv2.imwrite(outpath,img)
    cv2.imshow('Lena',img)
    cv2.waitKey(0)#To bind cv2 with keyboard event it will destroy all the windows if pressed
    cv2.destroyWindow('Lena')


# In[40]:


#NUMPY FOR IMAGES

def main():
    imgpath="E:\\clg imp\\DTU PROJECT\\Open CV\\standard_test_images\\lena_color_256.tif"
    img1 = cv2.imread(imgpath, 0)
    
    print(img1)
    print(type(img1))#for type
    print(img1.dtype)
    
    print(img1.shape)
    print(img1.ndim)
    print(img1.size)

    


# In[53]:


#using Webcam
def main():
    cap=cv2.VideoCapture(0)#to use the webcam #associates python with webcam
    
    if cap.isOpened():#For checking webcam is opened or not
        ret,frame=cap.read()#read and store in variable
        print(ret)
        print(frame)
    
    else:
        ret=False
    
    img1=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    plt.imshow(img1)
    plt.title('Color Image RGB')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    
    cap.release()

        
    


# In[61]:


#Live Video Processing
def main():
    windowName="Live Video Feed"
    cap=cv2.VideoCapture(0)#to use the webcam #associates python with webcam
    
    if cap.isOpened():#For checking webcam is opened or not
        ret,frame=cap.read()#read and store in variable
        #print(ret)
        #print(frame)
    
    else:
        ret=False
    
    #img1=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
   
    
    #.................skeleton of video which processes Live video.......
    #when we process image without  any sort of delay continuosly then it  becomes video, if you process image in 29.7 frames/sec then it becomes video
    while ret:
        ret,frame=cap.read()
        
        output=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#converting color to gray
        cv2.imshow("Gray", output)#will show gray output
        cv2.imshow(windowName, frame)#shows output in color
        if cv2.waitKey(1) == 27:#when escape is pressed break
            break

    cv2.destroyAllWindows()    
    
    
    cap.release()


# In[78]:


#Live Video Processing #Adjusting Resolution
def main():
    windowName="Live Video Feed"
    cap=cv2.VideoCapture(0)#to use the webcam #associates python with webcam
    
    #checking the current width and height of the window
    
    print("--------Before Updating Resolution-----")
    print('Width : ' + str(cap.get(3)))#Property 3 will return width
    print('Height : ' + str(cap.get(4)))#Property 4 will return Height
    
    cap.set(3, 1024)#set width to 1024
    cap.set(4, 768)#set height to 768
    
    #cap.set(3, 1800)#set width to 1024
    #cap.set(4, 900)#set height to 768

    
    print("--------After Updating Resolution-----")
    print('Width : ' + str(cap.get(3)))#Property 3 will return width
    print('Height : ' + str(cap.get(4)))
    #Note: it will update as per the resolution possible by the webcam as closest possible resolution

    
    if cap.isOpened():#For checking webcam is opened or not
        ret,frame=cap.read()#read and store in variable
        print(ret)
        print(frame)
    
    else:
        ret=False
    
    #img1=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    while ret:
        ret,frame=cap.read()
        
        output=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#converting color to gray
        cv2.imshow("Gray", output)#will show gray output
        cv2.imshow(windowName, frame)#shows output in color
        if cv2.waitKey(1) == 27:#when escape is pressed break
            break

    cv2.destroyAllWindows()    
    
    
    cap.release()


# In[92]:


#To write video to file
def main():
    
    windowName="Live Video Feed Capture"
    cv2.namedWindow(windowName)
    
    cap=cv2.VideoCapture(0)
    
    filename = 'E:\\clg imp\\DTU PROJECT\\Open CV\\output\\output.avi'
    
    codec = cv2.VideoWriter_fourcc('W', 'M', 'V', '2')
    #codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    #codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'J')
    #codec = cv2.VideoWriter_fourcc('W', 'M', 'V', '1')
    
    framerate = 30 #Framerate is needed to process images 
    resolution = (640, 480)#At lower resolution frame rate is going to be quite smoother
    
    VideoFileOutput = cv2.VideoWriter(filename, codec, framerate, resolution)
    
    
    if cap.isOpened():#For checking webcam is opened or not
        ret,frame=cap.read()#read and store in variable
        
    
    else:
        ret=False
    
   
    
    while ret:
        ret,frame=cap.read()
        
       
        
        #frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)#changes color to blue
        
        VideoFileOutput.write(frame)
        
        cv2.imshow(windowName, frame)
        
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()  
    VideoFileOutput.release()
    
    
    cap.release()


# In[10]:


#opencv video PLayer
def main():
    windowName = "OpenCV Video Player"
    cv2.namedWindow(windowName)
    
    filename = 'E:\\clg imp\\DTU PROJECT\\Open CV\\output\\output.avi'
    
    cap = cv2.VideoCapture(filename)
    
   
    #as long as file is open read from file and show
    while (cap.isOpened()): 
        
        ret, frame = cap.read()
        
        print(ret)#prints value of ret it will be either True or False
        
        if ret:
            cv2.imshow(windowName, frame)#if ret is true then show video on screen if it is false then break
            #if cv2.waitKey(66) == 27:
            if cv2.waitKey(130) == 27: #using this we can slow down or speed up   
                break
        else:
            break

    cv2.destroyAllWindows()    
    cap.release()


# In[43]:


#object Tracking by color
def main():

    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False


    while ret:
        
        ret, frame = cap.read()
        
        
        # Blue Color #it will track blue color irrespective of the shade
        
        low = np.array([100, 50, 50])
        high = np.array([140, 255, 255])
        
        # Red Color
        low = np.array([140, 150, 0])
        high = np.array([180, 255, 255])
        
        # Green Color
        #low = np.array([40, 50, 50])
        #high = np.array([80, 255, 255])
        
        
        #opencv captures bgr but bgr is not good for tracking so use hsv,hsv stands for hue saturation value
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        #imagemsk returns binary matrix based on low value and high value #low and high tells dark and light color                  
        image_mask = cv2.inRange(hsv, low, high)
        print(image_mask)
        
        #logical anding of masked image with original image gives color tracking facility
        output = cv2.bitwise_and(frame, frame, mask = image_mask)
        
        cv2.imshow("ImageMask",image_mask)#display imagemask on screen                
        cv2.imshow("Original Webcam Feed", frame)
        cv2.imshow("Color Tracking", output)#displays color tracked
                        
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()  
    cap.release()


# In[51]:


def main():

    cap = cv2.VideoCapture(0)
    
    #if cap.isOpened():
        #ret, frame = cap.read()
     #   ret, img = cap.read()
    #else:
     #   ret = False


    while (1):
        

         _, img = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Blue Color 
        
        blue_low = np.array([100, 50, 50])
        blue_high = np.array([140, 255, 255])
        
        
         # Red Color
        red_low = np.array([140, 150, 0])
        red_high = np.array([180, 255, 255])
        
        
        # Green Color
        green_low = np.array([40, 50, 50])
        green_high = np.array([80, 255, 255])
        
        
        
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        #imagemaskforcolors
        #range for colors
        blue_mask=cv2.inRange(hsv, blue_low, blue_high)
        red_mask=cv2.inRange(hsv, red_low, red_high)
        green_mask=cv2.inRange(hsv, green_low, green_high)
        
        
        
        #morphological transformation
        
        kernal=np.ones((5,5),"uint8")
        
        
        blue_ker=cv2.dilate(blue_mask,kernal)
        output_blue = cv2.bitwise_and(img, frame, mask = blue_mask)
        
        red_ker=cv2.dilate(red_mask,kernal)
        output_red = cv2.bitwise_and(img, frame, mask = red_mask)
        
        green_ker=cv2.dilate(green_mask,kernal)
        output_green = cv2.bitwise_and(img, frame, mask = green_mask)
        
        
        #tracking the blue color
        
        (_,contours,hierarchy)=cv2.findContours(blue_ker,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        for pic,contour in enumerate(contours):
            
            area=cv2.contourArea(contour)
            
            if(area>300):
                
                x,y,w,h=cv2.boundingRect(contour)
                img=cv2.rectangle(img(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(img,"BLUE COLOR",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0))
                
        
        
        #tracking the green color
        
        (_,contours,hierarchy)=cv2.findContours(blue_ker,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
        for pic,contour in enumerate(contours):
            area=cv2.contourArea(contour)
            if(area>300):
                
                x,y,w,h=cv2.boundingRect(contour)
                img=cv2.rectangle(img(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(img,"GREEN COLOR",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0))
                
        
        
        
        #tracking the red color
         (_,contours,hierarchy)=cv2.findContours(red_ker,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
         for pic,contour in enumerate(contours):
                area=cv2.contourArea(contour)
                if(area>300):
                
                x,y,w,h=cv2.boundingRect(contour)
                img=cv2.rectangle(img(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(img,"RED COLOR",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255))
                
        
                
                
        
        #cv2.imshow("ImageMask",image_mask)#display imagemask on screen                
        #cv2.imshow("Original Webcam Feed", frame)
        cv2.imshow("Color Tracking", img)#displays color tracked
        cv2.imshow("Color Tracking1", output_blue)
                        
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()  
    cap.release()
        


# In[54]:


import cv2   
import numpy as np

def main():
#capturing video through webcam
    cap=cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
    
    ret, frame = cap.read()

    while(ret):
        ret, frame = cap.read()

        #converting frame(img i.e BGR) to HSV (hue-saturation-value)

        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        #definig the range of red color
        red_lower=np.array([136,87,111],np.uint8)
        red_upper=np.array([180,255,255],np.uint8)

        #defining the Range of Blue color
        blue_lower=np.array([100, 50, 50],np.uint8)
        blue_upper=np.array([140, 100, 100],np.uint8)

        #defining the Range of yellow color
        yellow_lower=np.array([22,60,200],np.uint8)
        yellow_upper=np.array([60,255,255],np.uint8)

        #defining the Range of green color
        green_lower=np.array([22,60,200],np.uint8)
        green_upper=np.array([60,255,255],np.uint8)
        
        #finding the range of red,blue and yellow color in the image
        red=cv2.inRange(hsv, red_lower, red_upper)
        blue=cv2.inRange(hsv,blue_lower,blue_upper)
        yellow=cv2.inRange(hsv,yellow_lower,yellow_upper)
        green=cv2.inRange(hsv,green_lower,green_upper)
        
        #Morphological transformation, Dilation
        #kernal = np.ones((5 ,5), "uint8")

        red=cv2.dilate(red, np.ones((5 ,5), "uint8"),iterations=1)
        res=cv2.bitwise_and(frame,frame, mask = red)

        blue=cv2.dilate(blue,np.ones((5 ,5), "uint8"),iterations=1)
        res1=cv2.bitwise_and(frame,frame, mask = blue)

        yellow=cv2.dilate(yellow,np.ones((5 ,5), "uint8"),iterations=1)
        res2=cv2.bitwise_and(frame,frame, mask = yellow)    


        #Tracking the Red Color
        (_,contours,hierarchy)=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area>300):
                x,y,w,h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(frame,"RED color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))

        #Tracking the Blue Color
        (_,contours,hierarchy)=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area>300):
                x,y,w,h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(frame,"Blue color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0))

        #Tracking the yellow Color
        (_,contours,hierarchy)=cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area>300):
                x,y,w,h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,"yellow  color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))  


        #cv2.imshow("Redcolour",red)
        cv2.imshow("Color Tracking",frame)
        #cv2.imshow("red",res)
        if cv2.waitKey(1) ==27:
            cap.release()
            cv2.destroyAllWindows()
            break  
if __name__ == "__main__":
    main()


# In[46]:


if __name__=="__main__":
    main()
    

