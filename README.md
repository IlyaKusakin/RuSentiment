### Datasets
[From Corus](http://study.mokoron.com/), [RuSentiment](https://gitlab.com/kensand/rusentiment/tree/master/Dataset)

### RuBERT model
[rubert-base-cased](https://huggingface.co/DeepPavlov/rubert-base-cased)

[Package on Google Drive with weights and datasets](https://drive.google.com/drive/folders/1TZsdyYxAEU0Arrm-AQCFs-uNrXgznope?usp=sharing)


# Results

## stock RuBERT:

* Corus + RuSentiment dataset: 
  
  ROC AUC 0.78
  
  F1 0.80

* only RuSentiment dataset: 

  ROC AUC 0.75   
  
  F1 0.714

## pretrained RuBERT:

* Corus + RuSentiment dataset: 

  ROC AUC 0.9   
  
  F1 0.87

* only RuSentiment dataset: 

  ROC AUC 0.81   
  
  F1 0.78

## FastText:

* Corus + RuSentiment dataset: 

  ROC AUC 0.86   
  
  F1 0.77

* only RuSentiment dataset: 

  ROC AUC 0.81   
  
  F1 0.8
