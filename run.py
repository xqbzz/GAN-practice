import torch
from test import Generator
import matplotlib.pyplot as plt

# Загрузка сохраненных весов модели генератора
generator = Generator()
generator.load_state_dict(torch.load("generator.pth"))
generator.eval()

# Ввод текстового запроса пользователем
text_input = input("Введите описание для генерации изображения: ")

# Преобразование текстового запроса в тензор
latent_dim = 100
text_tensor = torch.randn(1, latent_dim)

# Генерация изображения
with torch.no_grad():
    generated_image = generator(text_tensor)

# Преобразование тензора в изображение для визуализации
generated_image = generated_image.squeeze().permute(1, 2, 0).numpy()

# Отображение сгенерированного изображения
plt.imshow(generated_image)
plt.axis("off")
plt.show()
