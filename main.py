import os
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog
from UI import Ui_MainWindow
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
from pytorch_model_summary import summary
from torchvision.models import vgg
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torch.nn as nn
import pickle
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
imagefile= []
imagefiletmp=[]
folder_path = ""
q1_1img=[]
imagePoints = []
objectPoints = []
Dist_all=[]
index = 1
q1_5img=[]
check=0
q2_1img=[]
q2_2img=[]
q2_folder_name = ""
q5_2img=[]
q5_1img=[]
name=[]
width =11
high = 8
generic_object_points = np.zeros((width * high, 3), np.float32)
generic_object_points[:, :2] = np.mgrid[0:width, 0:high].T.reshape(-1, 2)

class VGG_w_cls(nn.Module):
    def __init__(self):
        super(VGG_w_cls, self).__init__()
        self.m = nn.Sequential(
            vgg.vgg19_bn(),
            nn.Linear(1000, 10)
        )

    def forward(self, x):
        return self.m(x)
BATCH_SIZE = 32
EPOCH = 100
LR = 1e-3
labels = ["airplane", "car", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG_w_cls()
#best_model_pth, _ = QFileDialog.getOpenFileName(None, 'Select a file', '', 'All Files (*);;Text Files (*.txt)')
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))

def Load_folder():
    global folder_path
    global imagefile
    global q2_folder_name
    imagefile=[]
    options = QFileDialog.Options()
    folder_path = QFileDialog.getExistingDirectory(None, 'Select a folder', '', options=options)
    imagefiletmp = os.listdir(folder_path)
    for name in imagefiletmp:
        if ".bmp" in name:
            imagefile.append(name)
        else:
            q2_folder_name = name
    imagefile = [os.path.splitext(file)[0] for file in imagefile]
    imagefile = [int(file) for file in imagefile]
    imagefile.sort()
    imagefile = [str(file) + '.bmp' for file in imagefile]
    
    if folder_path:
        print('Selected Folder:', folder_path)
        print(imagefile)

def Find_corners():
    print("Start Find_corners")
    width = 11
    high = 8
    global q1_1img
    if len(q1_1img)!=15:
        for i in range(len(imagefile)):
            tmp = folder_path + '/'+imagefile[i]
            print(tmp)
            image = cv2.imread(tmp)
            grayimg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #cv2.imshow("image",grayimg)
            ret , corners = cv2.findChessboardCorners(grayimg, (width, high), None)
            print(ret)
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                cv2.cornerSubPix(grayimg, corners, (5, 5), (-1, -1), criteria)
                cv2.drawChessboardCorners(image, (width, high), corners, ret)
                q1_1img.append(image)
                #cv2.imshow('Chessboard Corners', image)
    for i in range(len(q1_1img)):
        print("start show")
        cv2.namedWindow("Find_corners",0)
        cv2.resizeWindow("Find_corners",512,512)
        cv2.imshow('Find_corners', q1_1img[i])
        cv2.waitKey(500)
    #cv2.findChessboardCorners(grayimg, (width, high), None)
    #cv2.cornerSubPix(image, corners, winSize, zeroZone, criteria)

def Find_intrinsic():
    global Dist_all
    #generic_object_points = np.zeros((width * high, 3), np.float32)
    #generic_object_points[:, :2] = np.mgrid[0:width, 0:high].T.reshape(-1, 2)
    for i in range(len(imagefile)):
        tmp = folder_path + '/'+imagefile[i]
        image = cv2.imread(tmp)
        grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(grayimg, (width, high), None)
        if ret:
            objectPoints.append(generic_object_points) 
            imagePoints.append(corners)
        image_shape = grayimg.shape[::-1]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, image_shape, None, None)
        #print(i,":")
        #print(objectPoints)
        Dist_all.append(dist)
        if ret:
            print("Intrinsic:")
            print(mtx)
        else:
            print("False")
    #cv2.calibrateCamera (objectPoints, imagePoints,(width, high) ,None, None)
def Selection_change():
    global index
    index = ui.ComboBox.currentText()
    print("change:",index)
    global check
    check+=1
    print("check",check)

def Find_extrinsic():
    global index
    #generic_object_points = np.zeros((width * high, 3), np.float32)
    #generic_object_points[:, :2] = np.mgrid[0:width, 0:high].T.reshape(-1, 2)
    if check == 0 :
        index = 1
    #print(type(index))
    #print(index)
    index = str(index)
    #print(type(index))
    tmp = folder_path + '/'+index+'.bmp'
    #print (tmp)
    image = cv2.imread(tmp)
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(grayimg, (width, high), None)
    if ret:
        objectPoints.append(generic_object_points)  
        imagePoints.append(corners)
        image_shape = grayimg.shape[::-1]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, image_shape, None, None)
    numpy_object_points = np.array(generic_object_points)
    numpy_image_points = np.array(corners)
    numpy_obp = np.array(objectPoints)
    numpy_imp = np.array(imagePoints)
    numpy_mtx = np.array(mtx)
    numpy_dist = np.array(dist)
    retval, rvec, tvec = cv2.solvePnP(numpy_object_points, numpy_image_points,numpy_mtx, numpy_dist)
    #retval, rvec, tvec = cv2.solvePnP(numpy_obp, numpy_imp,numpy_mtx, numpy_dist)
    R = cv2.Rodrigues(rvec)[0]
    extrinsic_matrix = np.hstack((R, tvec))
    print("extrinsic:")
    print(extrinsic_matrix)
    #print(dist)

def Find_distortion():
    if len(Dist_all)==15:
        print("Distortion:")
        for i in range(len(Dist_all)):
            print(Dist_all[i])
    else:
        #generic_object_points = np.zeros((width * high, 3), np.float32)
        #generic_object_points[:, :2] = np.mgrid[0:width, 0:high].T.reshape(-1, 2)
        print("Distortion:")
        for i in range(len(imagefile)):
            tmp = folder_path + '/'+imagefile[i]
            image = cv2.imread(tmp)
            grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(grayimg, (width, high), None)
            if ret:
                objectPoints.append(generic_object_points)  
                imagePoints.append(corners)
                image_shape = grayimg.shape[::-1]
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, image_shape, None, None)   
                print(dist)

def Show_result():
    global q1_5img
    #generic_object_points = np.zeros((width * high, 3), np.float32)
    #generic_object_points[:, :2] = np.mgrid[0:width, 0:high].T.reshape(-1, 2)
    for i in range(len(imagefile)):
        tmp = folder_path + '/'+imagefile[i]
        image = cv2.imread(tmp)
        h,w=image.shape[:2]
        grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(grayimg, (width, high), None)
        if ret:
            objectPoints.append(generic_object_points) 
            imagePoints.append(corners)
        image_shape = grayimg.shape[::-1]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, image_shape, None, None)
        np_mtx=np.array(mtx)
        np_dist=np.array(dist)
        undistorted_image = cv2.undistort(image, np_mtx, np_dist)
        tmp_img=np.zeros((h,w+w,3),dtype="uint8")
        tmp_img[0:h,0:w]=image
        tmp_img[0:h,w:]=undistorted_image
        q1_5img.append(tmp_img)
    for i in range(len(q1_5img)):
        cv2.namedWindow("result",0)
        cv2.resizeWindow("result",1024,512)
        cv2.imshow("result",q1_5img[i])
        cv2.waitKey(300)
def show_on_board():
    letter = ui.textEdit.toPlainText()
    letter = letter.upper()
    #generic_object_points = np.zeros((width * high, 3), np.float32)
    #generic_object_points[:, :2] = np.mgrid[0:width, 0:high].T.reshape(-1, 2)
    filename = folder_path+'/'+q2_folder_name+'/alphabet_lib_onboard.txt'
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    print(filename)
    for i in range(len(imagefile)):
        tmp = folder_path + '/'+imagefile[i]
        image = cv2.imread(tmp)
        grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(grayimg, (width, high), None)
        if ret:
            objectPoints.append(generic_object_points)  
            imagePoints.append(corners)
        image_shape = grayimg.shape[::-1]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, image_shape, None, None)
        np_dist=np.array(dist)
        np_mtx = np.array(mtx)
        numpy_object_points = np.array(generic_object_points)
        numpy_image_points = np.array(corners)
        retval, rvec, tvec = cv2.solvePnP(numpy_object_points, numpy_image_points,np_mtx, np_dist)
        R = cv2.Rodrigues(rvec)[0]
        extrinsic_matrix = np.hstack((R, tvec))
        np_rvec=np.array(rvec)
        np_tvec=np.array(tvec)
        x=2
        y=5
        for j in range(len(letter)):
            ch = fs.getNode(letter[j]).mat() 
            ch += np.array([x*3+1,y,0])
            x=x-1
            if x == -1:
                x=2
                y=2
            ch = ch.reshape((-1, 3)).astype(float)
            point,_ = cv2.projectPoints(ch,np_rvec,np_tvec,np_mtx,np_dist)
            point = point.reshape((-1,2,2)).astype(int)
            for k in point:
                cv2.line(image,k[0],k[1],(0,0,255),10)
        q2_2img.append(image)
    for i in range(len(q2_2img)):
        cv2.namedWindow("show words on a board",0)
        cv2.resizeWindow("show words on a board",512,512)
        cv2.imshow("show words on a board",q2_2img[i])
        cv2.waitKey(300)

def show_vertical():
    letter = ui.textEdit.toPlainText()
    letter = letter.upper()
    #generic_object_points = np.zeros((width * high, 3), np.float32)
    #generic_object_points[:, :2] = np.mgrid[0:width, 0:high].T.reshape(-1, 2)
    filename = folder_path+'/'+q2_folder_name+'/alphabet_lib_vertical.txt'
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    print(filename)
    for i in range(len(imagefile)):
        tmp = folder_path + '/'+imagefile[i]
        image = cv2.imread(tmp)
        grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(grayimg, (width, high), None)
        if ret:
            objectPoints.append(generic_object_points)  
            imagePoints.append(corners)
        image_shape = grayimg.shape[::-1]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, image_shape, None, None)
        np_dist=np.array(dist)
        np_mtx = np.array(mtx)
        numpy_object_points = np.array(generic_object_points)
        numpy_image_points = np.array(corners)
        retval, rvec, tvec = cv2.solvePnP(numpy_object_points, numpy_image_points,np_mtx, np_dist)
        R = cv2.Rodrigues(rvec)[0]
        extrinsic_matrix = np.hstack((R, tvec))
        np_rvec=np.array(rvec)
        np_tvec=np.array(tvec)
        x=2
        y=5
        for j in range(len(letter)):
            ch = fs.getNode(letter[j]).mat() 
            ch += np.array([x*3+1,y,0])
            x=x-1
            if x == -1:
                x=2
                y=2
            ch = ch.reshape((-1, 3)).astype(float)
            point,_ = cv2.projectPoints(ch,np_rvec,np_tvec,np_mtx,np_dist)
            point = point.reshape((-1,2,2)).astype(int)
            for k in point:
                cv2.line(image,k[0],k[1],(0,0,255),10)
        q2_1img.append(image)
    for i in range(len(q2_1img)):
        cv2.namedWindow("show words vertical",0)
        cv2.resizeWindow("show words vertical",512,512)
        cv2.imshow("show words vertical",q2_1img[i])
        cv2.waitKey(300)

def Load_Image_L():
    global imageL 
    file_name, _ = QFileDialog.getOpenFileName(None, 'Select a file', '', 'All Files (*);;Text Files (*.txt)')
    imageL = cv2.imread(file_name)
    print(file_name)

def Load_Image_R():
    global imageR
    file_name, _ = QFileDialog.getOpenFileName(None, 'Select a file', '', 'All Files (*);;Text Files (*.txt)')
    imageR = cv2.imread(file_name)
    print(file_name)

def Mouse_call(event,x,y,flags,param):
    disparity = Disparity_map[y][x]
    imageR_R = imageR.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(imageR_R,(x-disparity,y),10,(0,255,0),20)
        cv2.imshow('imageR',imageR_R)

def Stereo_Disparity_map():
    print(imageL)
    print(imageR)
    stereo = cv2.StereoBM_create(256,25)
    cv2.namedWindow('imageL',0)
    cv2.namedWindow('imageR',0)
    cv2.resizeWindow('imageL',512,512)
    cv2.resizeWindow('imageR',512,512)
    cv2.imshow('imageL',imageL)
    cv2.imshow('imageR',imageR)
    gray_imgL = cv2.cvtColor(imageL,cv2.COLOR_BGR2GRAY)
    gray_imgR = cv2.cvtColor(imageR,cv2.COLOR_BGR2GRAY)
    global Disparity_map
    Disparity_map = stereo.compute(gray_imgL,gray_imgR)
    Disparity_map = cv2.normalize(Disparity_map, Disparity_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.namedWindow('Disparity Map',0)
    cv2.resizeWindow('Disparity Map',512,512)
    cv2.imshow('Disparity Map',Disparity_map)
    cv2.setMouseCallback('imageL',Mouse_call)

def Load_Image1():
    global Left
    file_name, _ = QFileDialog.getOpenFileName(None, 'Select a file', '', 'All Files (*);;Text Files (*.txt)')
    Left=cv2.imread(file_name)
    print(file_name)

def Keypoints():
    #I1 = Left.copy()
    sift = cv2.SIFT_create()
    Left1 = cv2.cvtColor(Left,cv2.COLOR_BGR2GRAY)
    Left_keypoints, Left_descriptors = sift.detectAndCompute(Left1, None)
    I1 = cv2.drawKeypoints(Left1, Left_keypoints, None,color=(0,255,0))
    cv2.namedWindow('Left',0)
    cv2.resizeWindow('Left',512,512)
    cv2.namedWindow('I1',0)
    cv2.resizeWindow('I1',512,512)
    cv2.imshow('Left', Left)
    cv2.imshow('I1', I1)

def Load_Image_2():
    global Right
    file_name, _ = QFileDialog.getOpenFileName(None, 'Select a file', '', 'All Files (*);;Text Files (*.txt)')
    Right=cv2.imread(file_name)
    print(file_name)

def Matched_Keypoints():
    Left1 = cv2.cvtColor(Left,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    Right1 = cv2.cvtColor(Right,cv2.COLOR_BGR2GRAY)
    Left_keypoints, Left_descriptors = sift.detectAndCompute(Left1, None)
    Right_keypoints, Right_descriptors = sift.detectAndCompute(Right1, None)
    R_Keypoints = cv2.drawKeypoints(Right1, Right_keypoints, None,color=(0,255,0))
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(Left_descriptors,Right_descriptors,k=2)
    good=[]
    for m,n in matches:
        if m.distance<0.75*n.distance:
            good.append([m])
    I2= cv2.drawMatchesKnn(Left1,Left_keypoints,Right1,Right_keypoints,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #I2 = cv2.cvtColor(I2tmp,cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('I2',0)
    cv2.resizeWindow('I2',1024,512)
    cv2.imshow('I2',I2)
    
def Load_Image():
    global file_img
    file_img, _ = QFileDialog.getOpenFileName(None, 'Select a file', '', 'Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)')
    
    if not file_img:
        print("No file selected.")
        return
    
    print("Selected file:", file_img)
    tmp = f'<img src="{file_img}" width="128" height="128"/>'
    ui.Inference_Image.setHtml(tmp)


def Show_Augmented_Images():
    options = QFileDialog.Options()
    Q5_2folder = QFileDialog.getExistingDirectory(None, 'Select a folder', '', options=options)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30)
    ])    
    imagefiletmp = os.listdir(Q5_2folder)
    for i in range(len(imagefiletmp)):
        if '.png' in imagefiletmp[i]:
            q5_2img.append(Q5_2folder+'/'+imagefiletmp[i])
            name.append(imagefiletmp[i])

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, imag in enumerate(q5_2img):
        imag = Image.open(imag)
        imag = imag.convert('RGB')
        fin_imag = transform(imag)
        q5_1img.append(fin_imag)
        ax = axes.flatten()[i]
        ax.imshow(fin_imag)
        ax.set_title(name[i])  
        ax.axis('off')
    plt.tight_layout()
    plt.show()    

def Show_Model_Structure():
    result = "Model Structure\n"
    result += summary(VGG_w_cls(), torch.zeros((1, 3, 32, 32)), max_depth=2)
    print(result)

def Show_Acc_and_Loss():
    with open('train_loss.pkl', 'rb') as fp:
        train_loss = pickle.load(fp)
    with open('train_acc.pkl', 'rb') as fp:
        train_acc = pickle.load(fp)
    with open('test_loss.pkl', 'rb') as fp:
        test_loss = pickle.load(fp)
    with open('test_acc.pkl', 'rb') as fp:
        test_acc = pickle.load(fp)        
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.plot(train_loss,label='train_loss')
    plt.plot(test_loss,label='test_loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.ylabel("Acc (%)")
    plt.xlabel("epoch")
    plt.plot(train_acc,label='train_acc')
    plt.plot(test_acc,label='test_acc')
    plt.legend()
    plt.show()    

def Inference():
    transform = Compose([
            Resize((32, 32)), 
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    img = Image.open(file_img)
    img = transform(img).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        predict = model(img)
    prob = torch.nn.functional.softmax(predict,1)
    best_prob,best_catid = torch.max(prob,1)
    result = labels[best_catid]
    print(result)
    ui.label_Predict.setText(f'Predict ={result}')
    plt.figure(figsize=(10, 10))
    plt.bar(labels, prob.numpy()[0], color='skyblue')
    plt.ylabel('Probability')
    plt.xlabel('Class')
    plt.title('Probability of each class')
    plt.ylim(0, 1)
    plt.show()   

if __name__ == "__main__":
    cv2.setNumThreads(0)
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    global ui
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.Load_folder.clicked.connect(Load_folder)
    ui.Find_corners.clicked.connect(Find_corners)
    ui.Find_intrinsic.clicked.connect(Find_intrinsic)
    ui.ComboBox.activated.connect(Selection_change)
    ui.Find_extrinsic.clicked.connect(Find_extrinsic)
    ui.Find_distortion.clicked.connect(Find_distortion)
    ui.Show_result.clicked.connect(Show_result)
    ui.show_on_board.clicked.connect(show_on_board)
    ui.show_vertical.clicked.connect(show_vertical)
    ui.Load_Image_L.clicked.connect(Load_Image_L)
    ui.Load_Image_R.clicked.connect(Load_Image_R)
    ui.Stereo_map.clicked.connect(Stereo_Disparity_map)
    ui.Load_Image1.clicked.connect(Load_Image1)
    ui.Keypoints.clicked.connect(Keypoints)
    ui.Load_Image2.clicked.connect(Load_Image_2)
    ui.Matched_Keypoints.clicked.connect(Matched_Keypoints)
    ui.Load_Image_2.clicked.connect(Load_Image)
    ui.Show_Agumented_Images.clicked.connect(Show_Augmented_Images)
    ui.Show_Model_Structure.clicked.connect(Show_Model_Structure)
    ui.Show_Acc_and_Loss.clicked.connect(Show_Acc_and_Loss)
    ui.Inference.clicked.connect(Inference)
    MainWindow.show()
    sys.exit(app.exec_())
