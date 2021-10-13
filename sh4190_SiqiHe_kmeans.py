import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster


def UNI_Name_kmeans(imgPath,imgFilename,savedImgPath,savedImgFilename,k):
    """parameters:
    imgPath: the path of the image folder. Please use relative path
    imgFilename: the name of the image file
    savedImgPath: the path of the folder you will save the image
    savedImgFilename: the name of the output image
    k: the number of clusters of the k-means function
    function: using k-means to segment the image and save the result to an image with a bounding box"""
    # import image file
    comp_path= os.path.join(imgPath, imgFilename)
    img_org = cv2.imread(comp_path)
    img_RGB = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)

    #k-means function here
    vectorized = img_RGB.reshape((-1, 3)) #reshape image data
    kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=1).fit(vectorized) #run Kmeans function
    label = kmeans.labels_ #label for each value
    center = np.uint8(kmeans.cluster_centers_) #center for each cluaster and make sure values are integer
    result = center[label.flatten()] #segmented image data
    result_image = result.reshape((img_RGB.shape)) #reshape segmented image data for display
    #plot original image and after kmeans image
    plt.figure(1)
    plt.subplot(1, 2, 1), plt.imshow(img_RGB)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(result_image)
    plt.title('Segmented Image when K = %i' % k), plt.xticks([]), plt.yticks([])
    plt.show()

    #Find the right cluster: look for the face cluster
    #cluster_arr to store clusters for different centers respectively　
    cluster_arr = []
    for i in range(k):
        arr = result.copy()
        arr[label.flatten() != i] = 0 #When label != i, let the value = 0 (Only keep values with same label)
        cluster_arr.append(arr)
    #The RGB color for pale skin
    skin = [209, 163, 164]
    #Check which center is close to the pale skin
    #list to store the absolute difference between centers and pale skin for each center
    list = []
    for i in range(len(center)):
        sum = abs(np.sum(center[i] - skin))
        list.append(sum)
    #min value means this center is closet to the skin color
    #rihgtk is the index of min value in list which is the face center.
    rightk=np.argmin(list)
    #Convert this cluster image to a binary image (face and background)
    image = cluster_arr[rightk].reshape((img_RGB.shape)) #face color
    img_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)  # Gray image
    _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # Binary image
    #plot binary image
    plt.figure(2)
    plt.imshow(img_binary)
    plt.show()
    #Find the contours
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #The max area for contours bounded is the face (To exclude noise areas)
    max_area = 0
    cnt=0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > max_area:
            cnt = contours[i]
            max_area = area
    #Draw bounding box
    x, y, w, h = cv2.boundingRect(cnt) #find x,y coordinate and width/height of bounding rectangle
    rect_image = cv2.rectangle(img_RGB.copy(), (x, y), (x + w, y + h), (255, 0, 0), 2)#draw rectangle
    #plot figure with bounding box
    plt.figure(3)
    plt.imshow(rect_image)
    plt.show()
    #Plot the final image with a bounding box
    rect_image_print = cv2.cvtColor(rect_image, cv2.COLOR_BGR2RGB)
    #Save image file
    savedExt = ".jpg"
    filename = os.path.join(savedImgPath, savedImgFilename)
    plt.imsave(filename + savedExt, rect_image_print)
    cv2.imwrite(filename + savedExt, rect_image_print)


if __name__ == "__main__":
    imgPath="" #Write your own path
    imgFilename="face_d2.jpg"
    savedImgPath=""
    savedImgFilename="rectImg"
    k=5
    UNI_Name_kmeans(imgPath, imgFilename, savedImgPath, savedImgFilename, k)




