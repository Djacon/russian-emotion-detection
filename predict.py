from model import predict_emotion, predict_emotions


try:
    while True:
        text = input('Введите текст: ')
        emotion = predict_emotion(text)
        print('Бот: ваша эмоция -', emotion)
except KeyboardInterrupt:
    print('\n\nХорошего вам дня!')
