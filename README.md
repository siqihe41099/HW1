# HW1
### UNI_Name_kmeans(imgPath,imgFilename,savedImgPath,savedImgFilename,k)
#### Specify and Explain ALLkey variables/parameters used in the code. 
##### Input Parameters:
* imgPath: the path of the image folder. Please use relative path
* imgFilename: the name of the image file
* savedImgPath: the path of the folder you will save the image
* savedImgFilename: the name of the output image
* k: the number of clusters of the k-means function \
\
The variables except input parameters appeared in the code are explained in the script
##### Output:
* One plot of the comparsion between original image and segmented image (not be saved,just for display)
* One plot of the original image with a bounding box on the face (not be saved,just for display)
* One plot of the original image with a bounding box on the face (saved directly to the specified path)

### The logic behind the Flow:
I write this part in the script in detail. So Here I introduce it briefly. 
1. Import image and turn it into RGB
2. Apply kmeans to segment image and plot the segmented image
3. Find face cluster
   * compare each center of cluster to RGB of skin color and the index of min value is the right center
   * Convert the image into a binary image (face and background)
4. Find contour and display image with a bounding box
5. Save image to specified directory

### Limitations:
* It cannot recognize people for different color skin, since I manually set the skin color for pale skin. And 
  this choice change the result of bounding box. For example, I used [236, 188, 180] as skin color before, then
  only k=4,5,6 works. When k=7 to 10 it mistakenly bound the shirt. Then I change to [209, 163, 164], k = 4 to 10 
  works, even though k = 8 to 10 cannot bound the whole face region, but only half of the face. So the skin color matters.
* It need the correct k input. And in my program, we can only try different k manully. It is better to find a way 
  to look for best k rather than manually doing it.
* It cannot recognize multiple faces in one image.

### References
* kmeans: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html and 
  https://www.kdnuggets.com/2019/08/introduction-image-segmentation-k-means-clustering.html
* Skin Color Value: https://colorswall.com/palette/2513
* cv2.findContours: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
* cv2.boundingRect: https://www.pythonpool.com/cv2-boundingrect/
