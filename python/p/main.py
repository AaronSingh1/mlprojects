import cv2
import numpy as np
import utlis 

########################
#PARAMETERS
path="1.jpeg"#PATH OF IMAGE AS STORED IN THE FILE
widthImg=700#WIDTH OF IMAGE USED FOR RESIZING THE ACTUAL IMAGE
heightImg=700#HEIGHT OF IMAGE USED FOR RESIZING THE ACTUAL IMAGE
questions=5#NUMBER OF QUESTIONS DEFINED
choices=5#NUMBER OF CHOICES DEFINED
ans=[1,2,0,1,4]#ANSWER KEY
webcamFeed = True#The line of code webcamFeed = True suggests that a webcam feed is being utilized in your project. Setting webcamFeed to True typically indicates that you want to access and process live video from a connected webcam.
cameraNo=0#default webcam index ELSE 1 for external cam
#######################

cap=cv2.VideoCapture(cameraNo)
cap.set(10,150)

while True:
    if webcamFeed:success,img=cap.read()
    else:img=cv2.imread (path)


    #PROCESSING
    img=cv2.resize(img,(widthImg,heightImg))
    imgContours=img.copy()
    imgFinal = img.copy()
    imgBiggestcontours = img.copy()
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur((imgGray),(5,5),1)
    imgCanny=cv2.Canny(imgBlur,10,50)
    
    try:
        #FINDING ALL CONTOURS
        contours,hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours,contours,-1,(0,255,0),10)
        rectCon=utlis.rectContour(contours)
        biggestContour=utlis.getCornerPoints(rectCon[0])
        gradePoints = utlis.getCornerPoints(rectCon [1])
        #print(biggestContour)


        if biggestContour.size != 0 and gradePoints.size != 0:
            cv2.drawContours(imgBiggestcontours,biggestContour,-1,(0,255,0),20)
            cv2.drawContours(imgBiggestcontours, gradePoints,-1, (255, 0, 0), 20)
            biggestContour=utlis.reorder(biggestContour)
            gradePoints=utlis.reorder(gradePoints)

            pt1 = np.float32(biggestContour)
            pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
            matrix = cv2.getPerspectiveTransform(pt1,pt2)
            imgWarpColored = cv2.warpPerspective(img,matrix,(widthImg,heightImg))

            ptG1 = np.float32(gradePoints)
            ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))

            imgWarpGray=cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
            imgThresh=cv2.threshold(imgWarpGray,150,255,cv2.THRESH_BINARY_INV)[1]

            boxes = utlis.splitBoxes(imgThresh)

            #GETTING NON ZERO PIXEL VALUES OF EACH BOX
            myPixelVal=np.zeros((questions,choices))
            countC = 0
            countR = 0
            for image in boxes:
                totalPixels=cv2.countNonZero(image)
                myPixelVal[countR][countC]=totalPixels
                countC+=1
                if(countC==choices):countR +=1 ;countC=0

            #FINDING INDEX VALUES OF THE MARKINGS
            myIndex = []
            for x in range (0,questions):
                arr=myPixelVal[x]
                #print("arr",arr)
                myIndexVal = np.where(arr==np.amax(arr))
                #print(myIndexVal[0])
                myIndex.append(myIndexVal[0][0])

            #GRADING
            grading=[]
            for x in range (0,questions):
                if ans[x]==myIndex[x]:
                    grading.append(1)
                else:grading.append(0)

            score= (sum(grading)/questions)*100 #FINAL_SCORE
            print(score)

            #DISPLAYING ANSWERS
            imgResult = imgWarpColored.copy()
            imgResult = utlis.showAnswers(imgResult,myIndex,grading,ans,questions,choices)
            imRawDrawing = np.zeros_like(imgWarpColored)
            imRawDrawing = utlis.showAnswers(imRawDrawing, myIndex, grading, ans, questions, choices)
            invmatrix = cv2.getPerspectiveTransform(pt2, pt1)
            imgInvWarp = cv2.warpPerspective(imRawDrawing, invmatrix, (widthImg, heightImg))
            imgRawGrade=np.zeros_like(imgGradeDisplay)
            cv2.putText(imgRawGrade,str(int(score))+"%", (60,100), cv2.FONT_HERSHEY_COMPLEX,3,(0,255 ,255),3)#This line adds the text representing the score to the imgRawGrade image using the cv2.putText() function. The score is converted to an integer and displayed at position (60, 100) with a specific font, size, and color.
            #cv2.imshow("Grade",imgRawGrade)
            invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))

            imgFinal = cv2.addWeighted(imgFinal,1,imgInvWarp,1,0)
            imgFinal = cv2.addWeighted(imgFinal, 1,imgInvGradeDisplay, 1, 0)




        imgBlank=np.zeros_like(img)
        imageArray = ([img,imgGray,imgBlur,imgCanny],
                      [imgContours,imgBiggestcontours,imgWarpColored,imgThresh],
                      [imgResult,imRawDrawing,imgInvWarp, imgFinal])
    except:
        imgBlank = np.zeros_like(img)
        imageArray = ([img, imgGray, imgBlur, imgCanny],
                      [imgBlank,imgBlank, imgBlank, imgBlank],
                      [imgBlank,imgBlank, imgBlank, imgBlank])


    lables = [["Original","Gray","Blur","Canny"],
               ["Contours","Biggest Con","Warp","Threshold"],
               ["Result","Raw Drawing","Inv Warp","Final"]]
    imgStacked=utlis.stackImages(imageArray,0.3)
    cv2.imshow("Final Result",imgFinal)
    cv2.imshow("stacked images",imgStacked)
    if cv2.waitKey(1) & 0xFF==ord('s'):
        cv2.imwrite("FinalResult.jpg",imgFinal)
        cv2.waitKey(300)