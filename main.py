import torch
from transformers import BertForSequenceClassification, AutoTokenizer

LABELS = ['neutral', 'joy', 'sadness', 'anger', 'enthusiasm', 'surprise', 'disgust', 'fear', 'guilt', 'shame']
LABELS_RU = ['нейтрально', 'радость', 'грусть', 'гнев', 'интерес', 'удивление', 'отвращение', 'страх', 'вина', 'стыд']

model = BertForSequenceClassification.from_pretrained('Djacon/rubert-tiny2-russian-emotion-detection')
tokenizer = AutoTokenizer.from_pretrained('Djacon/rubert-tiny2-russian-emotion-detection')


# Predicting emotion in text
# Example: predict_emotion("Сегодня такой замечательный день!") -> Joy
@torch.no_grad()
def predict_emotion(text: str, labels: list = LABELS) -> str:
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors='pt')
    inputs = inputs.to(model.device)

    outputs = model(**inputs)

    pred = torch.nn.functional.sigmoid(outputs.logits)
    pred = pred.argmax(dim=1)

    return labels[pred[0]].title()


# Probabilistic prediction of emotion in a text
# Example: predict_emotions("Сегодня такой замечательный день!") ->
# -> {'neutral': 0.229, 'joy': 0.873, 'sadness': 0.045,...}
@torch.no_grad()
def predict_emotions(text: str, labels: list = LABELS) -> dict:
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors='pt')
    inputs = inputs.to(model.device)

    outputs = model(**inputs)

    pred = torch.nn.functional.sigmoid(outputs.logits)

    emotions_list = {}
    for i in range(len(pred[0].tolist())):
        emotions_list[labels[i]] = round(pred[0].tolist()[i], 3)
    return emotions_list


def main():
    try:
        while True:
            text = input('Enter Text (`q` for quit): ')
            if not text:
                continue
            elif text == 'q':
                return print('Bye 👋')
            print('Your emotion is:', predict_emotion(text))
    except KeyboardInterrupt:
        print('\nBye 👋')


if __name__ == '__main__':
    main()