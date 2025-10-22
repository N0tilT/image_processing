import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_img(img, title="Image", figsize=(8, 6)):
    """Функция для отображения одного изображения"""
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'morphology_output/{title}.png', dpi=300, bbox_inches='tight')
    plt.show()

def draw_multiple_images(images, titles, rows, cols, figsize=(15, 10), image_title="multiple"):
    """Функция для отображения нескольких изображений"""
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'morphology_output/{image_title}.png', dpi=300, bbox_inches='tight')
    plt.show()

img = cv2.imread('data/test.jpg')
if img is None:
    print("Ошибка: Не удалось загрузить изображение 'data/test.jpg'")
    exit()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=== Базовые морфологические операции ===")

print("\n1. Эрозия с разными структурными элементами:")

kernel_cross_3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
print("Крестообразное ядро 3x3:\n", kernel_cross_3)

img_erosion_cross_3 = cv2.morphologyEx(img_gray, cv2.MORPH_ERODE, kernel_cross_3)
draw_img(img_erosion_cross_3, "Эрозия (CROSS 3x3)")

kernel_ellipse_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
print("Эллиптическое ядро 5x5:\n", kernel_ellipse_5)

img_erosion_ellipse_5 = cv2.morphologyEx(img_gray, cv2.MORPH_ERODE, kernel_ellipse_5)
draw_img(img_erosion_ellipse_5, "Эрозия (ELLIPSE 5x5)")

print("\n2. Дилатация:")
kernel_cross_3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
img_dilation = cv2.morphologyEx(img_gray, cv2.MORPH_DILATE, kernel_cross_3)
draw_img(img_dilation, "Дилатация (CROSS 3x3)")

print("\n3. Top-hat преобразование:")
kernel_cross_6 = cv2.getStructuringElement(cv2.MORPH_CROSS, (6,6))
img_tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel_cross_6)
draw_img(img_tophat, "Top-hat (CROSS 6x6)")

print("\n4. Комплексные морфологические операции:")

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (6,6))

img_open = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
img_close = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
img_tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
img_blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

images_complex = [img_gray, img_open, img_close, img_tophat, img_blackhat]
titles_complex = [
    "Исходное изображение", 
    "Открытие (OPEN)", 
    "Закрытие (CLOSE)", 
    "Top-hat", 
    "Black-hat"
]
draw_multiple_images(images_complex, titles_complex, 2, 3, (16, 10),"complex")

print("\n=== Операция Hit-or-Miss ===")

input_image = np.array((
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 255, 255, 255, 0, 0, 0, 0, 255, 0],
    [0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0],
    [0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 255, 0, 0, 0, 255, 255, 0, 0],
    [0, 0, 255, 255, 255, 0, 0, 0, 255, 0, 0],
    [0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), dtype="uint8")

kernel_1 = np.array((
    [0,   1, 0],
    [-1,  1, 1],
    [-1, -1, 0]), dtype="int")

kernel_2 = np.array((
    [0, -1, -1],
    [1,  1, -1],
    [0,  1,  0]), dtype="int")

kernel_3 = np.array((
    [-1, -1, 0],
    [-1,  1, 1],
    [0,   1, 0]), dtype="int")

kernel_4 = np.array((
    [0,  1,  0],
    [1,  1, -1],
    [0, -1, -1]), dtype="int")

output_image_1 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel_1)
output_image_2 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel_2)
output_image_3 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel_3)
output_image_4 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel_4)

output_image = output_image_1 | output_image_2 | output_image_3 | output_image_4

rate = 50
input_image_large = cv2.resize(input_image, None, fx=rate, fy=rate, interpolation=cv2.INTER_NEAREST)
output_image_large = cv2.resize(output_image, None, fx=rate, fy=rate, interpolation=cv2.INTER_NEAREST)

images_hitmiss = [input_image_large, output_image_large]
titles_hitmiss = ["Исходное бинарное изображение", "Результат Hit-or-Miss"]
draw_multiple_images(images_hitmiss, titles_hitmiss, 1, 2, (12, 6),"hit-or-miss")

print("\nАнализ результатов:")
print("1. Эрозия - уменьшает объекты, убирает шум")
print("2. Дилатация - увеличивает объекты, заполняет пробелы")
print("3. Открытие - эрозия + дилатация, убирает мелкие объекты")
print("4. Закрытие - дилатация + эрозия, заполняет мелкие отверстия")
print("5. Top-hat - выделяет светлые объекты на темном фоне")
print("6. Black-hat - выделяет темные объекты на светлом фоне")
print("7. Hit-or-Miss - поиск специфических шаблонов в изображении")

print("\nОбработка завершена!")