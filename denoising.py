import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

def calculate_mse(img1, img2):
    """Вычисление среднеквадратичной ошибки"""
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

def calculate_psnr(img1, img2):
    """Вычисление пикового отношения сигнал-шум"""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def generate_salt_pepper_noise(img, noise_percent):
    """Генерация шума 'соль-перец'"""
    height, width, channels = img.shape
    mask = np.ones((height, width, channels))
    mask_size = height * width
    
    num_zeros = int(mask_size * noise_percent)
    
    for channel in range(channels):
        indices_to_change = np.random.choice(mask_size, num_zeros, replace=False)
        mask[indices_to_change % height, indices_to_change // height, channel] = 0
    
    return (img * mask).astype(np.uint8), mask

def display_comparison(images, titles, rows, cols, figsize=(15, 10),image_title="comparison"):
    """Функция для отображения сравнения изображений"""
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'denoising_output/{image_title}.png', dpi=300, bbox_inches='tight')
    plt.show()

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
        plt.savefig(f'denoising_output/{save_name}', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    img = cv2.imread('data/test.jpg')
    if img is None:
        print("Ошибка: Не удалось загрузить изображение 'data/test.jpg'")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print("=== Часть 1: Сжатие JPEG и оценка качества ===")
    
    flag, encoded_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    print(f"Сжатие успешно: {flag}")
    print(f"Размер до сжатия: {img.size} байт")
    print(f"Размер после сжатия: {encoded_img.size} байт")
    
    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    decoded_img_rgb = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
    
    mse = calculate_mse(img_rgb, decoded_img_rgb)
    psnr = calculate_psnr(img_rgb, decoded_img_rgb)
    print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    print(f"Пиковое отношение сигнал-шум (PSNR): {psnr:.2f} dB")
    
    images = [img_rgb, decoded_img_rgb]
    titles = ['Оригинал', f'Сжатое JPEG (PSNR: {psnr:.2f} dB)']
    display_comparison(images, titles, 1, 2, (12, 6),"compare_with_clean")
    
    print("\n=== Часть 2: Non-Local Means Denoising ===")
    
    noisy_img_gaussian = random_noise(img_rgb / 255.0, mode="gaussian") * 255.0
    noisy_img_gaussian = np.clip(noisy_img_gaussian, 0, 255).astype(np.uint8)
    noisy_img_gaussian_bgr = cv2.cvtColor(noisy_img_gaussian, cv2.COLOR_RGB2BGR)
    
    denoised_nlm = cv2.fastNlMeansDenoisingColored(
        noisy_img_gaussian_bgr, None, h=10, hColor=10, 
        templateWindowSize=7, searchWindowSize=21
    )
    denoised_nlm_rgb = cv2.cvtColor(denoised_nlm, cv2.COLOR_BGR2RGB)
    
    psnr_noisy = calculate_psnr(img_rgb, noisy_img_gaussian)
    psnr_denoised = calculate_psnr(img_rgb, denoised_nlm_rgb)
    print(f"PSNR между оригиналом и зашумленным изображением: {psnr_noisy:.2f} dB")
    print(f"PSNR между оригиналом и очищенным изображением: {psnr_denoised:.2f} dB")
    
    images = [noisy_img_gaussian, denoised_nlm_rgb]
    titles = [
        f'Гауссовский шум (PSNR: {psnr_noisy:.2f} dB)', 
        f'Non-Local Means (PSNR: {psnr_denoised:.2f} dB)'
    ]
    display_comparison(images, titles, 1, 2, (12, 6), "compressing")
    
    print("\n=== Часть 3: Обработка шума 'соль-перец' ===")
    
    noisy_img_salt_pepper, _ = generate_salt_pepper_noise(img_rgb, 0.1)
    noisy_img_salt_pepper_bgr = cv2.cvtColor(noisy_img_salt_pepper, cv2.COLOR_RGB2BGR)
    
    denoised_nlm_sp = cv2.fastNlMeansDenoisingColored(
        noisy_img_salt_pepper_bgr, None, h=10, hColor=10,
        templateWindowSize=7, searchWindowSize=21
    )
    denoised_nlm_sp_rgb = cv2.cvtColor(denoised_nlm_sp, cv2.COLOR_BGR2RGB)
    
    denoised_median = cv2.medianBlur(noisy_img_salt_pepper_bgr, 5)
    denoised_median_rgb = cv2.cvtColor(denoised_median, cv2.COLOR_BGR2RGB)
    
    psnr_noisy_sp = calculate_psnr(img_rgb, noisy_img_salt_pepper)
    psnr_nlm_sp = calculate_psnr(img_rgb, denoised_nlm_sp_rgb)
    psnr_median = calculate_psnr(img_rgb, denoised_median_rgb)
    
    print(f"PSNR зашумленное 'соль-перец': {psnr_noisy_sp:.2f} dB")
    print(f"PSNR Non-Local Means: {psnr_nlm_sp:.2f} dB")
    print(f"PSNR Медианный фильтр: {psnr_median:.2f} dB")
    
    images = [
        noisy_img_salt_pepper, 
        denoised_nlm_sp_rgb, 
        denoised_median_rgb
    ]
    titles = [
        f'Шум "соль-перец" (PSNR: {psnr_noisy_sp:.2f} dB)',
        f'Non-Local Means (PSNR: {psnr_nlm_sp:.2f} dB)',
        f'Медианный фильтр (PSNR: {psnr_median:.2f} dB)'
    ]
    display_comparison(images, titles, 1, 3, (15, 5),"noise")
    
    print("\n=== Сравнение методов денойзинга ===")
    
    methods = ["Гауссовский шум", "Non-Local Means", "Шум 'соль-перец'", "NLM для 'соль-перец'", "Медианный фильтр"]
    psnr_values = [psnr_noisy, psnr_denoised, psnr_noisy_sp, psnr_nlm_sp, psnr_median]
    
    print("\nСравнение методов по PSNR:")
    for method, psnr_val in zip(methods, psnr_values):
        print(f"{method}: {psnr_val:.2f} dB")
    
    all_images = [
        img_rgb, noisy_img_gaussian, denoised_nlm_rgb,
        noisy_img_salt_pepper, denoised_nlm_sp_rgb, denoised_median_rgb
    ]
    all_titles = [
        'Оригинал',
        f'Гауссовский шум\n({psnr_noisy:.1f} dB)',
        f'Non-Local Means\n({psnr_denoised:.1f} dB)',
        f'Шум "соль-перец"\n({psnr_noisy_sp:.1f} dB)',
        f'NLM для "соль-перец"\n({psnr_nlm_sp:.1f} dB)',
        f'Медианный фильтр\n({psnr_median:.1f} dB)'
    ]
    
    plt.figure(figsize=(18, 12))
    for i, (image, title) in enumerate(zip(all_images, all_titles)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.title(title, fontsize=12)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'denoising_output/compare_with_original.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()