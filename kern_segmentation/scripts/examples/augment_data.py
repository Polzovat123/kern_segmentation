import argparse
import os
import cv2
from tifffile import tifffile

from preprocessing.Augmentator import Augmentator
from preprocessing.Preparer import Preparer

def is_valid_file(parser, arg):
    """
    Проверка, существует ли файл
    """
    if not os.path.isfile(arg):
        parser.error(f"The file {arg} does not exist.")
    else:
        return arg

def augment_and_save(input_data_path, input_label_path, output_folder, d_a):
    data = cv2.imread(input_data_path, 0)
    label = cv2.imread(input_label_path, cv2.IMREAD_GRAYSCALE)  # Используйте cv2.IMREAD_GRAYSCALE

    data_blur, sobelx, sobely = Preparer.calculate(data=data, kernel_size=3, sobel_kernel_size=3)
    augmented_data = Augmentator.augment_data(data_blur, sobelx, sobely, d_a=d_a)
    augmented_label = Augmentator.augment_label(label, d_a=d_a)

    augmented_label = [cv2.cvtColor(augmented_label[i], cv2.COLOR_BGR2GRAY) for i in range(len(augmented_label))]


    os.makedirs(output_folder, exist_ok=True)

    print(data.shape, label.shape)
    for i in range(len(augmented_data)):
        tifffile.imwrite(f'{output_folder}/data_{i}.tif', augmented_data[i])
        tifffile.imwrite(f'{output_folder}/label_{i}.tif', augmented_label[i])

def main():
    parser = argparse.ArgumentParser(description='Augment data and labels.')

    parser.add_argument('--input_data_path', type=lambda x: is_valid_file(parser, x), help='Путь до входных данных')
    parser.add_argument('--input_label_path', type=lambda x: is_valid_file(parser, x), help='Путь до меток')
    parser.add_argument('--output_folder', type=str, default='pictures/output/augment', help='Папка для сохранения результата')
    parser.add_argument('--d_a', type=int, default=100, help='Параметр для аугментации')

    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"Дополнительные аргументы, которые могли быть добавлены poetry run: {unknown}")

    if not args.input_data_path or not args.input_label_path:
        print("Необходимо указать путь до входных данных и меток.")
        return

    augment_and_save(args.input_data_path, args.input_label_path, args.output_folder, args.d_a)

if __name__ == "__main__":
    main()
