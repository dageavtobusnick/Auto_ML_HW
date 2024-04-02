import main
while True:
    prompt = input("Введите фразу для перевода:")
    if prompt:
        print(main.translate_phrase(prompt))