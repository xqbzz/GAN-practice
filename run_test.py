import subprocess
import re
import matplotlib.pyplot as plt
import torch
from test import Generator, Discriminator

generator = Generator()
discriminator = Discriminator()

def run_test():
    process = subprocess.Popen(["python", "test.py"], stdout=subprocess.PIPE, universal_newlines=True)

    epochs = []
    d_losses = []
    g_losses = []

    for line in iter(process.stdout.readline, ''):
        # Отображение прогресса обучения
        if "[Epoch" in line:
            match = re.search(r"\[Epoch (\d+)/\d+\] \[Batch \d+/\d+\] \[D loss: (.+?)\] \[G loss: (.+?)\]", line)
            if match:
                epoch = int(match.group(1))
                d_loss = float(match.group(2))
                g_loss = float(match.group(3))

                epochs.append(epoch)
                d_losses.append(d_loss)
                g_losses.append(g_loss)

                print(line.strip())

    process.stdout.close()
    process.wait()

    # Сохранение модели
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")

    # Вывод графика прогресса обучения
    plt.plot(epochs, d_losses, label="D loss")
    plt.plot(epochs, g_losses, label="G loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_test()
