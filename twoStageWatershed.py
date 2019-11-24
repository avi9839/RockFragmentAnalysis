import numpy as np
import cv2
import operator
import math
from random import randint
from matplotlib import pyplot as plt
import time
# 0.2989, 0.5870, 0.1140. for RGB->gray
def check(markers,r,c,frag):
    if r!=0 and c!=cols and markers[r-1][c-1] == frag:
        return True
    if r!=0 and markers[r-1][c] == frag:
        return True
    if r!=0 and c!= cols and markers[r-1][c+1] == frag:
        return True
    if c!=0 and markers[r][c-1] == frag:
        return True
    if c!=cols and markers[r][c+1] == frag:
        return True
    if r!=rows and c!= 0 and markers[r+1][c-1] == frag:
        return True
    if r!=rows and markers[r+1][c] == frag:
        return True
    if r!=rows and c!=cols and markers[r+1][c+1] == frag :
        return True
    else:
        return False

def getlist(img,markers,r,c,frag):
    lis = [[r,c]]
    markers[r][c] = -5
    flag = True
    while flag:
        if (c!=0 and markers[r][c - 1] == -1 and check(markers,r,c-1,frag)):
            c -= 1
        elif r!=rows and (markers[r + 1][c] == -1 and check(markers,r+1,c,frag)):
            r += 1
        elif (c!=cols and markers[r][c + 1] == -1 and check(markers,r,c+1,frag)):
            c += 1
        elif (r!=0 and markers[r - 1][c] == -1 and check(markers,r-1,c,frag)):
            r -= 1
        elif (r!=0 and c!=0 and markers[r - 1][c - 1] == -1 and check(markers, r - 1, c - 1, frag)):
            r -= 1
            c -= 1
        elif (r != rows and c!=0 and markers[r + 1][c - 1] == -1 and check(markers, r + 1, c - 1, frag)):
            r += 1
            c -= 1
        elif (r!=0 and c!=cols and markers[r - 1][c + 1] == -1 and check(markers,r-1,c+1,frag)):
            r -= 1
            c += 1
        elif (r!=rows and c!=cols and r != rows and c != cols and markers[r + 1][c + 1] == -1 and check(markers, r + 1, c + 1, frag)):
            r += 1
            c += 1
        else:
            flag = False
        markers[r][c]=-5
        lis.append([r,c])
        img[r][c]=[0,255,0]
    for i in lis:
        markers[i[0]][i[1]]=-1
    return lis

def angle(list,i,p):
    a = math.sqrt((list[i-p][0]-list[i+p][0])*(list[i-p][0]-list[i+p][0])+(list[i-p][1]-list[i+p][1])*(list[i-p][1]-list[i+p][1]))
    b = math.sqrt((list[i][0] - list[i + p][0]) * (list[i][0] - list[i + p][0]) + (list[i][1] - list[i + p][1]) * (list[i][1] - list[i + p][1]))
    c = math.sqrt((list[i - p][0] - list[i][0]) * (list[i - p][0] - list[i][0]) + (list[i - p][1] - list[i][1]) * (list[i - p][1] - list[i][1]))
    d = (b*b+c*c-a*a)/(2*b*c)
    if(d>1 or d<-1): #invalid cos value
        return 180
    return math.degrees(math.acos(d))

def fillColor(img,markers,x,y,ct,frag):
    if ct>993:
        return
    if(markers[x][y]==frag):
        markers[x][y]=163
    else:
        img[x][y]=[255,0,0]
        return
    fillColor(img, markers, x - 1, y, ct + 1,frag)
    fillColor(img, markers, x + 1, y, ct + 1,frag)
    fillColor(img, markers,x,y-1,ct+1,frag)
    fillColor(img,markers,x,y+1,ct+1,frag)
    return

def UtilJoinPairedPoints1(markers,l,p,x,y,img):
    for i in range(1,x+1):
        markers[l[p][0]-i][l[p][1]]=-1
        img[l[p][0]-i][l[p][1]]=[0,0,0]
    for i in range(1,y+1):
        markers[l[p][0]-x][l[p][1]-i]=-1
        img[l[p][0]-x][l[p][1]-i]=[0,0,0]
    return

def UtilJoinPairedPoints2(markers,l,p,x,y,img):
    x *= -1
    for i in range(1,x+1):
        markers[l[p][0]+i][l[p][1]]=-1
        img[l[p][0]+i][l[p][1]]=[0,0,0]
    for i in range(1,y+1):
        markers[l[p][0]+x][l[p][1]-i]=-1
        img[l[p][0]+x][l[p][1]-i]=[0,0,0]
    return

def UtilJoinPairedPoints3(markers,l,p,x,y,img):
    for i in range(1, x + 1):
        markers[l[p][0] - i][l[p][1]] = -1
        img[l[p][0] - i][l[p][1]]=[0,0,0]
    y *= -1
    for i in range(1, y + 1):
        markers[l[p][0] - x][l[p][1] + i] = -1
        img[l[p][0] - x][l[p][1] + i]=[0,0,0]
    return


def UtilJoinPairedPoints4(markers,l,p,x,y,img):
    x *= -1
    for i in range(1, x + 1):
        markers[l[p][0] + i][l[p][1]] = -1
        img[l[p][0] + i][l[p][1]]=[0,0,0]
    y *= -1
    for i in range(1, y + 1):
        markers[l[p][0] + x][l[p][1] + i] = -1
        img[l[p][0] + x][l[p][1] + i]=[0,0,0]
    return


def joinPairedPoints(markers,l,i,j,img):
    x = l[i][0]-l[j][0]
    y = l[i][1]-l[j][1]
    if x>0 and y>0:
        UtilJoinPairedPoints1(markers,l,i,x,y,img) # connect i,j points in the list l
    if x<0 and y>0:
        UtilJoinPairedPoints2(markers, l, i, x, y,img)
    if x>0 and y<0:
        UtilJoinPairedPoints3(markers,l,i,x,y,img)
    if x<0 and y<0:
        UtilJoinPairedPoints4(markers,l,i,x,y,img)
    return

def getDistance(l,i,j):
    return math.sqrt((l[i][0]-l[j][0])*(l[i][0]-l[j][0])+(l[i][1]-l[j][1])*(l[i][1]-l[j][1]))

def joinConcavePoints(img, markers, finallis): #join concave points
    count = len(finallis)
    for i in range(count):
        if finallis[i][3]!=0:
            p=0
            min=100000
            for j in range(count):
                if j != i:
                    d = getDistance(finallis,i,j)
                    if d<min:
                        min=d
                        p = j
            if(finallis[p][2]+finallis[i][2]>min/2):
                joinPairedPoints(markers,finallis,i,p,img)
                finallis[p][3]=0
                finallis[i][3]=0

# join the concave points
def join(markers,canny,img,x,y,frag,finallis):
    flag = True
    ct=0
    lis = []
    while flag: #check for points on the edge which have neighbouring point in the fragment and also an edge
        if (canny[x- 1][y - 1] == 255 and markers[x - 1][y - 1] == frag):
            x -= 1
            y -= 1
        elif (canny[x - 1][y] == 255 and markers[x - 1][y] == frag):
            x -= 1
        elif (canny[x - 1][y + 1] == 255 and markers[x - 1][y + 1] == frag):
            x -= 1
            y += 1
        elif (canny[x][y - 1] == 255 and markers[x][y-1] == frag):
            y -= 1
        elif (canny[x][y+ 1] == 255 and markers[x][y+ 1] == frag):
            y += 1
        elif (canny[x+ 1][y- 1] == 255 and markers[x+ 1][y- 1] == frag):
            x += 1
            y -= 1
        elif (canny[x+ 1][y] == 255 and markers[x+ 1][y] == frag):
            x += 1
        elif (canny[x+1][y+ 1] == 255 and markers[x+ 1][y+ 1] == frag):
            x += 1
            y += 1
        elif markers[x-1][y-1]==-1 or markers[x-1][y]==-1 or markers[x-1][y+1]==-1 or markers[x][y-1]==-1 or markers[x][y+1]==-1 or markers[x+1][y-1]==-1 or markers[x+1][y]==-1 or markers[x+1][y+1]==-1:
            if ct > 20:
                point = [x, y, ct, 1]
                #finallis.append(point)
                #markers[x + 1][y - 1]=-1
                #mask = np.zeros((602, 802), np.uint8)
                #cv2.floodFill(markers, mask, (x+10,y+10), 170)
                #fillColor(img,markers,x,y+1,0,frag)
            return
        else:
            point = [x, y, ct, 1]
            finallis.append(point)
            img[x][y]=[255,255,255]
            return
        markers[x][y]=-5
        canny[x][y]=-1
        img[x][y]=[0,0,0]
        lis.append([x,y])
        ct += 1
    for p in lis:
        markers[p[0]][p[1]]=-1
        img[p[0]][p[1]]=[255,0,0]

# check true cannny partition based on connection with canny edge and fragment region
def checkTruePartition(i, canny, markers, frag):
    if (i[0]!=0 and i[1]!=0 and canny[i[0] - 1][i[1] - 1] == 255 and markers[i[0] - 1][i[1] - 1] == frag):
        return True
    elif (i[0]!=0 and canny[i[0] - 1][i[1]] == 255 and markers[i[0] - 1][i[1]] == frag):
        return True
    elif (i[0]!=0 and i[1]!=cols and canny[i[0] - 1][i[1] + 1] == 255 and markers[i[0] - 1][i[1] + 1] == frag):
        return True
    elif (i[1]!=0 and canny[i[0]][i[1] - 1] == 255 and markers[i[0]][i[1] - 1] == frag):
        return True
    elif (i[1]!=cols and canny[i[0]][i[1] + 1] == 255 and markers[i[0]][i[1] + 1] == frag):
        return True
    elif (i[0]!=rows and i[1]!=0 and canny[i[0] + 1][i[1] - 1] == 255 and markers[i[0] + 1][i[1] - 1] == frag):
        return True
    elif (i[0]!=rows and canny[i[0] + 1][i[1]] == 255 and markers[i[0] + 1][i[1]] == frag):
        return True
    elif (i[0]!=rows and i[1]!=cols and canny[i[0] + 1][i[1] + 1] == 255 and markers[i[0] + 1][i[1] + 1] == frag):
        return True
    else:
        return False

def iterate(img,markers,canny,list,i,p,frag,newList): # i-point element in list, p-iteration, ct-count
    ang = angle(list,i,p)
    if(ang<130 and checkTruePartition(list[i],canny,markers,frag)):
        img[list[i][0]][list[i][1]] = [255, 0, 0]
        join(markers, canny, img, list[i][0], list[i][1], frag, newList)
    return

def getFraginInitial(markers,r,c):
    if(r!=0 and c!=0 and markers[r-1][c-1]!=-1 and markers[r-1][c-1]!=1 ):
        return markers[r-1][c-1]
    if (r!=0 and markers[r - 1][c] != -1 and markers[r - 1][c] != 1):
        return markers[r - 1][c]
    if (r!=0 and c!=cols and markers[r - 1][c + 1] != -1 and markers[r - 1][c + 1] != 1):
        return markers[r - 1][c + 1]
    if (c!=0 and markers[r][c - 1] != -1 and markers[r][c - 1] != 1):
        return markers[r][c - 1]
    if (c!=cols and markers[r][c + 1] != -1 and markers[r][c + 1] != 1):
        return markers[r][c + 1]
    if (r!=rows and c!=0 and markers[r + 1][c - 1] != -1 and markers[r + 1][c - 1] != 1):
        return markers[r + 1][c - 1]
    if (r!=rows and markers[r + 1][c] != -1 and markers[r + 1][c] != 1):
        return markers[r + 1][c]
    if (r!=rows and c!=cols and markers[r + 1][c + 1] != -1 and markers[r + 1][c + 1] != 1):
        return markers[r + 1][c + 1]
    else:
        return 0

def getlistandMerge(img,markers,baseMarkers,r,c,frag,perimeter):
    lis = [[r,c]]
    markers[r][c] = -5
    flag = True
    dic = {}
    prevFrag = 1
    global gcount
    while flag:
        if (c!=0 and markers[r][c - 1] == -1 and check(markers,r,c-1,frag)):
            c = c - 1
        elif r!=rows and (markers[r + 1][c] == -1 and check(markers,r+1,c,frag)==1):
             r = r + 1
        elif (c!=cols and markers[r][c + 1] == -1 and check(markers,r,c+1,frag)):
            c=c+1
        elif (r!=0 and markers[r - 1][c] == -1 and check(markers,r-1,c,frag)):
            r = r - 1
        elif (r!=0 and c!=0 and markers[r - 1][c - 1] == -1 and check(markers, r - 1, c - 1, frag) == 1):
            r = r - 1
            c = c - 1
        elif (r != rows and c!=0 and markers[r + 1][c - 1] == -1 and check(markers, r + 1, c - 1, frag) == 1):
            r = r + 1
            c = c - 1
        elif (r!=0 and c!=cols and markers[r - 1][c + 1] == -1 and check(markers,r-1,c+1,frag)==1):
            r = r - 1
            c = c + 1
        elif (r!=rows and c!=cols and r != rows and c != cols and markers[r + 1][c + 1] == -1 and check(markers, r + 1, c + 1, frag) == 1):
            r = r + 1
            c = c + 1
        else :
            flag = False
        markers[r][c]=-5
        lis.append([r,c])
        img[r][c]=[255,255,0]
        a = getFraginInitial(baseMarkers, r, c)  # get the neighbouring fragment in the first stage segmentation result
        if a in dic:
            if a != prevFrag:
                dic[a] = 1
            dic[a] += 1
        else:
            dic[a] = 1
        prevFrag = a

    dic[0] = 0
    flag = False
    if len(dic) > 0:
        k = max(dic, key=dic.get)
        if dic[k] > len(lis)/5:
            baseMarkers[markers == frag] = k # update the first watershed dictionary
            flag = True # Merging happened
        else:
            baseMarkers[markers==frag] = gcount # add new fragment to the base marker
            perimeter[gcount] = len(lis) # add new fragment perimeter
            gcount += 1
    cm=0
    if flag:
        for i in lis:
            markers[i[0]][i[1]] = -1
            if getFraginInitial(baseMarkers, i[0], i[1]) == k:
                cm += 1
                baseMarkers[i[0]][i[1]] = k
    else:
        for i in lis:
            markers[i[0]][i[1]] = -1
    # update perimeter length if merging happened
    if flag:
        perimeter[k] += (len(lis)-cm)

    return lis

def mergeImage(baseMarkers,markers,img,canny,perimeter): #basemarkers-firstround watershed marker, markers-second round watershed marker
    markers[markers == 0] = 1
    mat = markers
    mat = np.array(mat).reshape(1, mat.size)[0]
    mat = np.unique(mat, return_index=True, return_inverse=False, return_counts=False)
    pos = mat[1]
    count = len(pos)
    n = cols + 1
    newd = {}
    for i in range(2,count):
        newd[mat[0][i]] = int(pos[i]/n)
    print(newd)
    print("Total fragments = ", count)
    for frag in range(2, count):  # frag - fragment
        if frag in newd:
            r = newd[frag]
            for c in range(0, cols):
                value = markers[r][c]
                if value == -1 and (markers[r - 1][c - 1] == frag or markers[r + 1][c + 1] == frag or markers[r + 1][c] == frag or markers[r][c + 1] == frag or markers[r + 1][c - 1] == frag or markers[r - 1][c + 1] == frag or markers[r - 1][c] == frag or markers[r][c - 1] == frag):
                    # make list of all point on current fragment boundary
                    list = getlistandMerge(img, markers, baseMarkers,r, c, frag,perimeter)
                    newList = []  # list to store the concave points
                    for i in range(10, len(list) - 10):
                        x = (list[i - 10][0] + list[i + 10][0]) / 2
                        y = (list[i - 10][1] + list[i + 10][1]) / 2
                        if markers[x][y] != frag and markers[x][y] != -1:
                            iterate(img, markers, canny, list, i, 10,frag,newList)  # iterate to mark the concave points
                    if not newList:
                        joinConcavePoints(img, markers, newList)
                    break


def secondWatershed(baseMarkers,img,baseCanny,perimeter):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Conversion to gray scale
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # thresholding

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, 2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, marker = cv2.connectedComponents(sure_fg)
    marker += 1
    marker[unknown == 255] = 0
    cv2.watershed(img, marker)
    img[marker==-1]=[255,0,0]
    mergeImage(baseMarkers,marker,img,baseCanny,perimeter)

#if __name__ == "__main__":
t1 = time.time()
img = cv2.imread('img/52.jpg')
screen_res = 1080, 520
scale_width = screen_res[0] / img.shape[1]
scale_height = screen_res[1] / img.shape[0]
scale = min(scale_width, scale_height)
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)
rows = img.shape[0]-1  #total number of rows
cols = img.shape[1]-1  #total number of columns
'''cv2.namedWindow('Load Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Load Image', window_width, window_height)
cv2.imshow('Load Image',img)'''

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        #Conversion to gray scale
canny = cv2.Canny(gray,100,200)
#gray = cv2.GaussianBlur(gray,(3,3),0)

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)   #thresholding

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 5)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)
markers += 1
markers[unknown == 255] = 0
cv2.watershed(img, markers)
""" # Marker labelling
 marker[i][j] = -1, border line
 marker[i][j] = 1, background region
 marker[i][j] = (k : k!=1,-1 ), fragment no. """

'''for i in range(cols+1):
    img[0][i]=[0,0,0]
    img[rows][i]=[0,0,0]
    if markers[rows-1][i]==1:
        markers[rows][i]=1
for i in range(rows+1):
    img[i][0]=[0,0,0]
    img[i][cols]=[0,0,0]
img[markers==-1]=(0,0,255)'''

print('Image Size:',markers.shape)

markers[markers == 0] = 1
mat = markers
mat = np.array(mat).reshape(1,mat.size)[0]
mat = np.unique(mat,return_index=True, return_inverse=False, return_counts=True)
pos = mat[1]
count = len(pos)
n = cols+1
'''print(mat[0])
print(mat[2])
print(pos)'''
print("Total fragments =",count)

global gcount
gcount = count - 1

perimeter = {}
# get concave points in each fragment
for frag in range(2,count): # frag - fragment
            img[markers == frag] = [0,0,0]  # convert segmented rock into background
            r = int(pos[frag]/n)
            k = pos[frag]%n
            for c in range(k, cols):
                value = markers[r][c]
                if value == -1 and (markers[r-1][c-1] == frag or markers[r+1][c+1] == frag or markers[r+1][c] ==frag or markers[r][c+1] == frag or markers[r+1][c-1] == frag or markers[r-1][c+1] == frag or markers[r-1][c] == frag or markers[r][c-1] == frag):
                    lis = getlist(img,markers,r,c,frag) # make list of all point on current fragment boundary
                    newList = []  # list to store the concave points
                    perimeter[frag] = len(lis)
                    for i in range(10, len(lis) - 10):
                        x = (lis[i - 10][0] + lis[i + 10][0]) / 2
                        y = (lis[i - 10][1] + lis[i + 10][1]) / 2
                        if markers[x][y] != frag and markers[x][y]!=-1:
                            iterate(img, markers, canny, lis, i, 10, frag, newList) #iterate to mark the concave points
                    if not newList:
                        joinConcavePoints(img,markers,newList)
                    break

'''cv2.namedWindow('Watershed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Watershed', window_width, window_height)
cv2.imshow('Watershed', img)
cv2.imwrite('img.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

print('Perimeter', perimeter)

secondWatershed(markers,img,canny,perimeter) # second stage watershed segmentation
dic = {}
ct=0
print('peri',perimeter)
#dic = sorted(dic.items(), key=operator.itemgetter(1))

markers[markers == -1] = 0
markers[markers == -5] = 0

#count the size of each rock
data = np.bincount(markers.flatten())
count = len(data)

'''for i in range(2,count):
    a = randint(0, 255)
    b = randint(0, 255)
    c = randint(0, 255)
    img[markers==i]=[a,b,c]'''

print('Total fragments',count-1)

'''cv2.namedWindow('Watershed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Watershed', window_width, window_height)
cv2.imshow('Watershed', img)
cv2.imwrite('img.jpg',img)'''


lis = []
scale = 45.5
for i in range(2,count):
        lis.append((4*data[i])/(perimeter[i]*scale))

# f = list(map(lambda x,y: 4*x/y, data[2:], perimeter[2:]))
lis.sort()
n = len(lis)
p = 100/n
lis2 = []
for i in range(n):
    lis2.append((i+1)*p)

print(lis)
print(lis2)
t2 = time.time()
print('time',t2-t1)
lines = plt.plot(lis,lis2)
plt.setp(lines, color='r', linewidth=2.0)
plt.title('Fragmentation size Distribution')
plt.xlabel('Fragment no.(Total fragments: '+str(count-2)+')')
plt.ylabel('Fragment area')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
