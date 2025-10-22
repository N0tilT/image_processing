import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy

def create_output_dir():
    """Создает папку для сохранения результатов"""
    if not os.path.exists('contours_output'):
        os.makedirs('contours_output')

def read_rgb(path):
    """Чтение изображения в RGB формате"""
    image = cv2.imread(path)
    if image is None:
        print(f"Ошибка: Не удалось загрузить изображение {path}")
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def read_gray(path):
    """Чтение изображения в градациях серого"""
    image = cv2.imread(path)
    if image is None:
        print(f"Ошибка: Не удалось загрузить изображение {path}")
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def read_binary(path, thr=128):
    """Бинаризация изображения с предварительным размытием"""
    image = read_gray(path)
    if image is None:
        return None
    
    image_ = cv2.GaussianBlur(image, (7, 7), 0)
    _, binary = cv2.threshold(image_, thr, 255, cv2.THRESH_BINARY_INV)
    return binary

def save_image(image, title="Image", save_name=None):
    """Отображение изображения с возможностью сохранения"""
    plt.figure(figsize=(8, 6))
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    if save_name:
        plt.savefig(f'contours_output/{save_name}', dpi=300, bbox_inches='tight')
    plt.show()

def Approximating(contour, epsilon=0.2):
    """Аппроксимация контура полигоном"""
    length = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon * length, True)
    return approx

def four_point_transform(image, pts):
    """Трансформация перспективы по 4 точкам"""
    pts = pts.reshape(4, 2)
    
    def order_points(pts):
        """Упорядочивание точек по часовой стрелке"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  
        rect[2] = pts[np.argmax(s)]  
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  
        rect[3] = pts[np.argmax(diff)]  
        return rect

    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def main():
    """Основная функция для работы с контурами"""
    create_output_dir()
    
    print("=== Загрузка и предобработка изображения ===")
    rgb_image = read_rgb('data/test.jpg')
    if rgb_image is None:
        return
    
    save_image(rgb_image, "Исходное RGB изображение", "01_original_rgb.png")
    
    gray_image = read_gray('data/test.jpg')
    save_image(gray_image, "Изображение в градациях серого", "02_grayscale.png")
    
    print("\n=== Бинаризация ===")
    binary_image = read_binary('data/test.jpg', 240)
    save_image(binary_image, "Бинаризованное изображение", "03_binary.png")
    
    canny_img = cv2.Canny(gray_image, 100, 200)
    save_image(canny_img, "Границы Canny", "04_canny.png")
    
    print("\n=== Поиск контуров ===")
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Найдено контуров: {len(contours)}")
    print(f"Иерархия контуров: {hierarchy.shape}")
    
    rgb_copy = rgb_image.copy()
    cv2.drawContours(rgb_copy, contours, -1, (255, 0, 0), 2)
    save_image(rgb_copy, "Все найденные контуры", "05_all_contours.png")
    
    print("\n=== Аппроксимация контуров ===")
    approx = Approximating(contours[2], 0.05)
    
    rgb_copy = rgb_image.copy()
    cv2.drawContours(rgb_copy, [approx], -1, (255, 0, 0), 2)
    
    for point in approx:
        x = point[0][0]
        y = point[0][1]
        rgb_copy = cv2.circle(rgb_copy, (x, y), radius=5, color=(0, 0, 255), thickness=3)
    
    save_image(rgb_copy, "Аппроксимированный контур с углами", "06_approximated_contour.png")
    
    print("\n=== Трансформация перспективы ===")
    if len(approx) == 4:
        warped = four_point_transform(rgb_image, approx)
        save_image(warped, "Трансформированное изображение", "07_perspective_transform.png")
    
    print("\n=== Выпуклая оболочка ===")
    rgb_copy = rgb_image.copy()
    hull = cv2.convexHull(contours[2])
    cv2.drawContours(rgb_copy, [hull], -1, (0, 0, 255), 3)
    save_image(rgb_copy, "Выпуклая оболочка", "08_convex_hull.png")
    
    print("\n=== Ограничивающие прямоугольники ===")
    rgb_copy = rgb_image.copy()
    
    x, y, w, h = cv2.boundingRect(contours[3])
    cv2.rectangle(rgb_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    rect = cv2.minAreaRect(contours[3])
    box = cv2.boxPoints(rect)
    box = np.array(box).astype(int)
    cv2.drawContours(rgb_copy, [box], 0, (0, 0, 255), 2)
    
    save_image(rgb_copy, "Ограничивающие прямоугольники", "09_bounding_boxes.png")

    print("\n=== Сортировка контуров по площади ===")
    areas = [cv2.contourArea(cnt) for cnt in contours]
    sorted_contours = sorted(zip(areas, contours), reverse=True)
    
    for i, (area, contour) in enumerate(sorted_contours[:3]):
        rgb_copy = rgb_image.copy()
        print(f"Контур {i}: Площадь = {area:.2f}")
        cv2.drawContours(rgb_copy, [contour], 0, (0, 255, 0), 3)
        save_image(rgb_copy, f"Контур {i} (Площадь: {area:.2f})", f"10_contour_{i}_area.png")
    
    print("\n=== Анализ принадлежности точек ===")
    rgb_copy = rgb_image.copy()
    
    point = (250, 250)
    cnt = sorted_contours[5][1] if len(sorted_contours) > 5 else sorted_contours[0][1]
    
    cnt = Approximating(cnt, 0.02)
    
    for corner in cnt:
        x = corner[0][0]
        y = corner[0][1]
        rgb_copy = cv2.circle(rgb_copy, (x, y), radius=5, color=(255, 0, 255), thickness=3)
    
    dist = cv2.pointPolygonTest(cnt, point, False)
    status = "внутри" if dist > 0 else "снаружи" if dist < 0 else "на границе"
    print(f"Точка {point} находится {status} контура (расстояние: {dist})")
    
    rgb_copy = cv2.circle(rgb_copy, point, radius=10, color=(255, 0, 0), thickness=3)
    cv2.drawContours(rgb_copy, [cnt], -1, (0, 255, 0), 5)
    
    save_image(rgb_copy, f"Анализ точки (расстояние: {dist:.2f})", "11_point_analysis.png")
    
    print("\n=== Экстремальные точки контура ===")
    rgb_copy = rgb_image.copy()
    
    cnt = sorted_contours[6][1] if len(sorted_contours) > 6 else sorted_contours[0][1]
    
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    
    points = [leftmost, rightmost, topmost, bottommost]
    colors = [(255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255)]
    
    for point, color in zip(points, colors):
        rgb_copy = cv2.circle(rgb_copy, point, radius=5, color=color, thickness=5)
    
    save_image(rgb_copy, "Экстремальные точки контура", "12_extreme_points.png")
    
    print("\nОбработка завершена! Результаты сохранены в папку 'contours_output'")

if __name__ == "__main__":
    main()