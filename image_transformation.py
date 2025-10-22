import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def display_images(images, titles, rows, cols, figsize=(15, 10),image_title="multiple"):
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
    plt.savefig(f'image_transformation/{image_title}.png', dpi=300, bbox_inches='tight')
    plt.show()

def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    """Функция для изменения яркости и контраста"""
    adjusted = image.astype(float) * contrast + brightness
    adjusted = np.clip(adjusted, 0, 255)
    return adjusted.astype(np.uint8)

print("=== ГЕОМЕТРИЧЕСКИЕ ПРЕОБРАЗОВАНИЯ ИЗОБРАЖЕНИЙ ===")

img = cv2.imread('data/test.jpg')
if img is None:
    print("Ошибка: Не удалось загрузить изображение 'data/test.jpg'")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(f"Original shape: {img.shape}")
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.title('Исходное изображение')
plt.axis('off')
plt.show()

print("\n=== 1. МАСШТАБИРОВАНИЕ ===")

res_up_cubic = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
res_up_linear = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

res_down_area = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
res_down_cubic = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)

images_scaling = [img, res_up_cubic, res_down_area, res_down_cubic]
titles_scaling = ['Original', 'Увеличение 2x (INTER_CUBIC)', 
                 'Уменьшение 0.3x (INTER_AREA)', 'Уменьшение 0.3x (INTER_CUBIC)']
display_images(images_scaling, titles_scaling, 2, 2,image_title="scale")

print("\n=== 2. СДВИГ ===")
rows, cols, _ = img.shape

M_translation = np.float32([[1, 0, 100], [0, 1, 50]])
dst_translation = cv2.warpAffine(img, M_translation, (cols, rows))

display_images([img, dst_translation], ['Original', 'Сдвиг (100,50)'], 1, 2,image_title="translation")

print("\n=== 3. ПОВОРОТ ===")

alpha = 45
scale = 1
center = ((cols-1)/2.0, (rows-1)/2.0)
M_rotation = cv2.getRotationMatrix2D(center, alpha, scale)
dst_rotation = cv2.warpAffine(img, M_rotation, (cols, rows))

alpha2 = -30
M_rotation2 = cv2.getRotationMatrix2D(center, alpha2, scale)
dst_rotation2 = cv2.warpAffine(img, M_rotation2, (cols, rows))

images_rotation = [img, dst_rotation, dst_rotation2]
titles_rotation = ['Original', 'Поворот +45°', 'Поворот -30°']
display_images(images_rotation, titles_rotation, 1, 3,image_title="rotate")

print("\n=== 4. АФФИННОЕ ПРЕОБРАЗОВАНИЕ ===")

pts1 = np.float32([[455, 180], [450, 100], [230, 180]])
pts2 = np.float32([[500, 100], [400, 50], [100, 300]])

img_with_points = img.copy()
for point in pts1:
    cv2.circle(img_with_points, tuple(point.astype(int)), 5, (0, 0, 255), -1)

M_affine = cv2.getAffineTransform(pts1, pts2)
dst_affine = cv2.warpAffine(img, M_affine, (cols, rows))

for point in pts2:
    cv2.circle(dst_affine, tuple(point.astype(int)), 5, (0, 0, 255), -1)

display_images([img_with_points, dst_affine], 
               ['Исходное с точками', 'Аффинное преобразование'], 1, 2,image_title="affine")

print("\n=== 5. МАСШТАБИРОВАНИЕ ЧЕРЕЗ АФФИННОЕ ПРЕОБРАЗОВАНИЕ ===")
alpha_scale = 2
beta_scale = 2
pts1_scale = np.float32([[455, 180], [450, 100], [230, 180]])
pts2_scale = np.float32([[455*alpha_scale, 180*beta_scale], 
                        [450*alpha_scale, 100*beta_scale], 
                        [230*alpha_scale, 180*beta_scale]])

M_affine_scale = cv2.getAffineTransform(pts1_scale, pts2_scale)
dst_affine_scale = cv2.warpAffine(img, M_affine_scale, 
                                 (cols*alpha_scale, rows*beta_scale), 
                                 cv2.INTER_LINEAR)

display_images([img, dst_affine_scale], 
               ['Original', f'Аффинное масштабирование ({alpha_scale}x)'], 1, 2,image_title="affine scale")

print("\n=== 6. ПЕРСПЕКТИВНОЕ ПРЕОБРАЗОВАНИЕ ===")

img_test2 = cv2.imread('data/test2.jpg')
if img_test2 is None:
    print("Предупреждение: Не удалось загрузить 'data/test2.jpg'. Пропускаем перспективное преобразование.")
else:
    rows_s, cols_s, ch = img_test2.shape
    pts1_perspective = np.float32([[56, 65], [368, 52], [28, 287], [389, 390]])
    pts2_perspective = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    M_perspective = cv2.getPerspectiveTransform(pts1_perspective, pts2_perspective)
    dst_perspective = cv2.warpPerspective(img_test2, M_perspective, (300, 300))

    img_test2_points = img_test2.copy()
    for point in pts1_perspective:
        cv2.circle(img_test2_points, tuple(point.astype(int)), 5, (0, 0, 255), -1)

    images_perspective = [img_test2_points, dst_perspective]
    titles_perspective = ['test2 с точками', 'Перспективное преобразование']
    display_images(images_perspective, titles_perspective, 1, 2, image_title="perspective translation")

print("\n=== 7. ИЗМЕНЕНИЕ ЯРКОСТИ И КОНТРАСТА ===")

bright_high = adjust_brightness_contrast(img, brightness=50, contrast=1.0) 
bright_low = adjust_brightness_contrast(img, brightness=-50, contrast=1.0) 
contrast_high = adjust_brightness_contrast(img, brightness=0, contrast=1.5) 
contrast_low = adjust_brightness_contrast(img, brightness=0, contrast=0.5)  
both_adjusted = adjust_brightness_contrast(img, brightness=30, contrast=1.3)

images_bc = [img, bright_high, bright_low, contrast_high, contrast_low, both_adjusted]
titles_bc = ['Original', 'Яркость +50', 'Яркость -50', 
             'Контраст x1.5', 'Контраст x0.5', 'Ярк. +30, Контр. x1.3']
display_images(images_bc, titles_bc, 2, 3,image_title="brightness")

print("\n=== 8. ФИЛЬТРАЦИЯ ИЗОБРАЖЕНИЙ ===")

blur_mean = cv2.blur(img, (5, 5))           
blur_gaussian = cv2.GaussianBlur(img, (5, 5), 0)  
blur_median = cv2.medianBlur(img, 5)        

kernel_sharpen = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
sharpened = cv2.filter2D(img, -1, kernel_sharpen)

bilateral = cv2.bilateralFilter(img, 9, 75, 75)

images_filters = [img, blur_mean, blur_gaussian, blur_median, sharpened, bilateral]
titles_filters = ['Original', 'Усредняющий фильтр', 'Гауссовский фильтр', 
                  'Медианный фильтр', 'Повышение резкости', 'Биллатеральный фильтр']
display_images(images_filters, titles_filters, 2, 3,image_title="filters")

print("\n=== ОБЪЯСНЕНИЕ ПРЕОБРАЗОВАНИЙ ===")
explanation = """
ПРЕОБРАЗОВАНИЯ ДЛЯ ПРЕДОБРАБОТКИ ИЗОБРАЖЕНИЙ:

Геометрические преобразования:
• Масштабирование - приведение изображений к единому размеру
• Поворот и сдвиг - аугментация данных, компенсация наклона камеры
• Аффинные преобразования - коррекция перспективных искажений

Изменение яркости и контраста:
• Нормализация освещения между разными снимками
• Улучшение видимости деталей в темных/светлых областях
• Подготовка для алгоритмов компьютерного зрения

Фильтрация:
• Удаление шума и артефактов
• Сглаживание для уменьшения влияния шума
• Повышение резкости для выделения важных деталей
• Сохранение границ при удалении шума (билинейная фильтрация)

Эти преобразования помогают улучшить качество входных данных для последующего анализа
нейронными сетями и классическими алгоритмами компьютерного зрения.
"""
print(explanation)

print("Обработка завершена!")