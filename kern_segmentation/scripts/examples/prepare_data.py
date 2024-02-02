import argparse
import os
import cv2
from matplotlib import pyplot as plt
from preprocessing.Preparer import Preparer

def is_valid_file(parser, arg):
    """
    Проверка, является ли файл существующим файлом
    """
    if not os.path.isfile(arg):
        parser.error(f"The file {arg} does not exist.")
    else:
        return arg

def main():
    parser = argparse.ArgumentParser(description='Process an image using Preparer module.')

    # Изменим аргументы для работы с poetry run
    parser.add_argument('--image_path', type=lambda x: is_valid_file(parser, x), help='Путь до входного изображения')
    parser.add_argument('--output_folder', type=str, default='output', help='Папка для сохранения результата')

    args, unknown = parser.parse_known_args()

    # Проверим наличие дополнительных аргументов, которые могли быть добавлены poetry run
    if unknown:
        print(f"Дополнительные аргументы, которые могли быть добавлены poetry run: {unknown}")

    # Проверим, что все обязательные аргументы переданы
    if not args.image_path:
        print("Необходимо указать путь до входного изображения.")
        return

    img = cv2.imread(args.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (data_blur, sobelx, sobely) = Preparer.calculate(data=img)

    # Абсолютный путь для сохранения графика
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_folder = os.path.join(project_path, 'pictures', 'output')
    os.makedirs(output_folder, exist_ok=True)

    # Путь для сохранения графика
    save_path = os.path.join(output_folder, 'prepare_data_result.png')

    # Сохраняем график
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex=True, sharey=True)
    im = axs[0].imshow(data_blur, cmap='gray')
    axs[0].set_title('Blur')
    plt.colorbar(im, ax=axs[0])
    im = axs[1].imshow(sobelx, cmap='gray')
    axs[1].set_title('Sobel X')
    plt.colorbar(im, ax=axs[1])
    im = axs[2].imshow(sobely, cmap='gray')
    axs[2].set_title('Sobel Y')
    plt.colorbar(im, ax=axs[2])
    plt.savefig(save_path)

    print(f"Результат сохранен в {save_path}")
    print(data_blur.shape, sobelx.shape, sobely.shape)

if __name__ == "__main__":
    main()
