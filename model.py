import torch
from transformers import BertForSequenceClassification, AutoTokenizer

model = BertForSequenceClassification.from_pretrained('emotion_detection_model')
tokenizer = AutoTokenizer.from_pretrained('emotion_detection_model')

LABELS = ['радость', 'интерес', 'удивление', 'печаль', 'гнев', 'отвращение',
          'страх', 'вина', 'нейтрально']


# Predicting emotion in text
@torch.no_grad()
def predict_emotion(text):
    inputs = tokenizer(text, truncation=True, return_tensors='pt')
    inputs = inputs.to(model.device)

    outputs = model(**inputs)

    pred = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = pred.argmax(dim=1)

    return LABELS[pred[0]]


# Probabilistic prediction of emotion in a text
@torch.no_grad()
def predict_emotions(text):
    inputs = tokenizer(text, truncation=True, return_tensors='pt')
    inputs = inputs.to(model.device)

    outputs = model(**inputs)

    pred = torch.nn.functional.softmax(outputs.logits, dim=1)

    emotions_list = {}
    for i in range(len(pred[0].tolist())):
        emotions_list[LABELS[i]] = round(pred[0].tolist()[i], 4)
    return emotions_list
