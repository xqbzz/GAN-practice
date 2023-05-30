import os

def rename_photos(folder_path):
    # Получаем список файлов в папке
    file_list = os.listdir(folder_path)
    # Фильтруем только файлы с расширениями изображений
    image_files = [f for f in file_list if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # Сортируем файлы по имени
    image_files.sort()

    # Переименовываем файлы
    for i, filename in enumerate(image_files):
        # Формируем новое имя файла в виде "числовой_порядок.расширение"
        new_name = str(i) + os.path.splitext(filename)[1]
        # Удаляем символы "_" из нового имени файла
        new_name = new_name.replace("_", "")
        # Создаем полный путь к старому файлу
        old_path = os.path.join(folder_path, filename)
        # Создаем полный путь к новому файлу
        new_path = os.path.join(folder_path, new_name)

        # Проверяем наличие файла с новым именем
        while os.path.exists(new_path):
            # Если файл с новым именем уже существует, добавляем "_1" к имени
            base_name, extension = os.path.splitext(new_name)
            new_name = base_name + "_1" + extension
            new_path = os.path.join(folder_path, new_name)

        # Переименовываем файл
        os.rename(old_path, new_path)

    print("Переименование завершено.")

# Задайте путь к папке с фотографиями
folder_path = "C:/Users/Сергей/PycharmProjects/GAN-practice/data/avatars"

# Вызываем функцию для переименования фотографий
rename_photos(folder_path)
