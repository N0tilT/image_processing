import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def segment_color(image, lower_bound, upper_bound):
    """Функция для сегментации по цвету в HSV пространстве"""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    return mask, segmented_image

def display_images(images, titles, rows, cols, figsize=(15, 10)):
    """Функция для отображения нескольких изображений"""
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i+1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else: 
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print("=== Часть 1: Сегментация синего цвета ===")
image1 = cv2.imread('data/test.jpg')
if image1 is None:
    print("Ошибка: Не удалось загрузить изображение 'data/test.jpg'")
else:
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    
    mask, segmented_image = segment_color(image1, lower_blue, upper_blue)
    
    images = [image1, mask, segmented_image]
    titles = ['Original Image', 'Mask', 'Segmented Image']
    display_images(images, titles, 1, 3, (12, 6))

print("\n=== Часть 2: Детекция кожи ===")
image2 = cv2.imread('data/test2.jpg')
if image2 is None:
    print("Ошибка: Не удалось загрузить изображение 'data/test2.jpg'")
else:
    img_HSV = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    img_YCrCb = cv2.cvtColor(image2, cv2.COLOR_BGR2YCrCb)
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
    global_mask = cv2.medianBlur(global_mask, 3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
    
    HSV_result = cv2.bitwise_and(image2, image2, mask=HSV_mask)
    YCrCb_result = cv2.bitwise_and(image2, image2, mask=YCrCb_mask)
    global_result = cv2.bitwise_and(image2, image2, mask=global_mask)
    
    img_resized = cv2.resize(image2, (640, 480))
    HSV_result = cv2.resize(HSV_result, (640, 480))
    YCrCb_result = cv2.resize(YCrCb_result, (640, 480))
    global_result = cv2.resize(global_result, (640, 480))
    
    images = [img_resized, HSV_result, YCrCb_result, global_result]
    titles = ['Original Image', 'HSV Detection', 'YCrCb Detection', 'Combined Detection']
    display_images(images, titles, 2, 2, (15, 10))

print("\n=== Часть 3: Поиск контуров ===")
if 'image2' in locals() and image2 is not None:
    img_contour1 = deepcopy(image2)
    _, thresh_img = cv2.threshold(img_contour1, 100, 255, cv2.THRESH_BINARY)
    
    channel = 0
    draw_color = [0, 0, 0]
    draw_color[channel] = 255
    contours, _ = cv2.findContours(thresh_img[:,:,channel], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_contour1, contours, -1, tuple(draw_color), 3)
    
    img_contour2 = deepcopy(image2)
    gray_img = cv2.cvtColor(img_contour2, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(img_contour2, cv2.COLOR_BGR2HSV)
    
    thr = 15
    _, thresh_img2 = cv2.threshold(hsv_img[:,:,0], thr, 180, cv2.THRESH_BINARY)
    contours2, _ = cv2.findContours(thresh_img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_contour2, contours2, -1, (0, 255, 0), 3)
    
    images = [thresh_img, img_contour1, thresh_img2, img_contour2]
    titles = ['Binary Threshold', 'Contours (Method 1)', 'Hue Threshold', 'Contours (Method 2)']
    display_images(images, titles, 2, 2, (15, 10))

print("\nОбработка завершена!")
cv2.imwrite('./colors_output/result_segmented.jpg', segmented_image)
cv2.imwrite('./colors_output/result_skin_detection.jpg', global_result)