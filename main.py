'''
choose four point in image and programm will automaticly do perspective transformation on image

also save the result image and matrix in folder



'''


import glob
import os.path

import numpy as np
import cv2  # freetype
from collections import deque
from matplotlib import pyplot as plt
global point_List
point_List = deque(maxlen=4)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect.astype(np.float32)

# 创建回调函数
def OnMouseAction(event, x, y, flags, param):
    img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 20, (10, 255, 10), -1)
        point_List.append((x, y))
        # cv2.imshow('image1', img)

        # print(point_List)


def read(img_path, mask_path):
    img1 = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    mask_name = os.path.basename(mask_path)
    img_name = os.path.basename(img_path)
    cv2.namedWindow('image1', 0)
    cv2.setMouseCallback('image1', OnMouseAction, img1)
    # cv2.imshow('image1', img1)
    # cv2.waitKey()
    img_copy = img1.copy()
    M=None
    while (1):
        # cv2.resizeWindow('image1', img1.shape[1] // 2, img1.shape[0] // 2)

        cv2.imshow('image1', img1)

        k = cv2.waitKey(1)
        if k == ord('q') or k== ord('Q'):

            cv2.imwrite(f'./test/{img_name}', img1)
            cv2.imwrite(f'./test1015/{mask_name}', mask)
            np.save(f'./testmask/{mask_name}', M)
            break
        elif k == ord('c') or k== ord('C'):
            img1 = img_copy.copy()
            cv2.setMouseCallback('image1', OnMouseAction, img1)
            point_List.clear()
        elif k == ord('t') or k== ord('T'):
            break
        elif k == ord('s') or k== ord('S'):
            if len(point_List)==4:
                sort_pt = order_points(np.array(point_List))
                print(sort_pt)
                # p = np.array(sort_pt[3])-np.array(sort_pt[0])
                # x,y=np.max(sort_pt,axis=0)-np.min(sort_pt,axis=0)

                (tl, tr, br, bl) = sort_pt
                # compute the width of the new image, which will be the
                # maximum distance between bottom-right and bottom-left
                # x-coordiates or the top-right and top-left x-coordinates
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))
                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))

                x = int(maxWidth)
                y = int(maxHeight)
                print(x,y)
                # pts2 = np.float32([[0, 0], [x, 0], [0, y], [x, y]])
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")

                M = cv2.getPerspectiveTransform(sort_pt, dst)
                mask = cv2.warpPerspective(mask, M, (maxWidth, maxHeight))
                img1 = cv2.warpPerspective(img_copy, M, (maxWidth, maxHeight))
                #
                mask = cv2.resize(mask,(640,480))
                img1 = cv2.resize(img1,(640,480))

            else:
                print(point_List)
    cv2.destroyAllWindows()

def check_mask(mask_path):
    print('-'*10,mask_path,'-'*10)
    mask = cv2.imread(mask_path)[...,0]
    plt.imshow(mask)
    plt.show()
def check_mask_pers(npy_path,mask_path):
    M =np.load(npy_path)
    print(M.shape)
    print('-'*10,mask_path,'-'*10)
    mask = cv2.imread(mask_path)[...,0]
    plt.imshow(mask)
    plt.show()
    mask = cv2.warpPerspective(mask, M, (640, 480))
    plt.imshow(mask)
    plt.show()
if __name__ == '__main__':
    imgs = glob.glob(r'F:\semantic_segmentation_unet\output_test\picked\2022_9_27_9_patient12_top.mp4\imgs\*.jpg')
    masks= glob.glob(r'F:\semantic_segmentation_unet\output_test\picked\2022_9_27_9_patient12_top.mp4\export\*.tiff')
    # npys= glob.glob(r'.\MA_test_masks\*.npy')
    # for n ,m in zip(npys,masks):
    #     check_mask_pers(n,m)
    for i,m in zip(imgs,masks):

        read(i,m)

