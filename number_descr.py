def number_descriptions():
    with open("descriptions.txt", "w", encoding="utf-8") as file:
        for i in range(153):
            line = f"Описание {i}: \n"
            file.write(line)

    print("Пронумерование описаний завершено.")

# Вызываем функцию для пронумерования описаний
number_descriptions()
