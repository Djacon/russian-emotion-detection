# Russian-Emotion-Detection

## Short Description

The __rubert-tiny2-russian-emotion-detection__ is a fine-tuned [rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2) model for multi-label __emotion classification__ task, specifically on Russian texts. Trained on custom [ru-izard-emotions](https://huggingface.co/datasets/Djacon/ru-izard-emotions) dataset, so this model can recognize a spectrum of 9 emotions, including __joy__, __sadness__, __anger__, __enthusiasm__, __surprise__, __disgust__, __fear__, __guilt__, __shame__ + __neutral__ (no emotion). Project was inspired by the [Izard's model](https://en.wikipedia.org/wiki/Differential_Emotions_Scale) of human emotions.

## Training Parameters:
```yaml
Optimizer: AdamW
Schedule: LambdaLR
Learning Rate: 1e-4
Batch Size: 64
Number Of Epochs: 10
```

## Emotion Categories:
```js
0. Neutral (Нейтрально)
1. Joy (Радость)
2. Sadness (Грусть)
3. Anger (Гнев)
4. Enthusiasm (Интерес)
5. Surprise (Удивление)
6. Disgust (Отвращение)
7. Fear (Страх)
8. Guilt (Вина)
9. Shame (Стыд)
```

## Test results:

||Neutral|Joy|Sadness|Anger|Enthusiasm|Surprise|Disgust|Fear|Guilt|Shame|Mean|
|-|-|-|-|-|-|-|-|-|-|-|-|
|AUC|0.7319|0.8234|0.8069|0.7884|0.8493|0.8047|0.8147|0.9034|0.8528|0.7145|0.8090|
|F1 micro|0.7192|0.7951|0.8204|0.7642|0.8630|0.9032|0.9156|0.9482|0.9526|0.9606|0.8642|
|F1 macro|0.6021|0.7237|0.6548|0.6274|0.7291|0.5712|0.4780|0.8158|0.4879|0.4900|0.6180|