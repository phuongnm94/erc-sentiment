## Emotions Relationship Modeling in the Conversation-level Sentiment Analysis  
### Abtract 
Sentiment analysis, also called opinion mining, is a task of Natural Language Processing (NLP) that aims to extract sentiments and opinions from texts. Among them, emotion recognition in conversation (ERC) is becoming increasingly popular as a new research topic in natural language processing (NLP). 
In this work, we investigate the dependencies between the emotional states among utterances in a conversation. Our intuition is that the emotional states can be affected or transferred between speakers in a conversation and do not limited by the length of the conversation. 
The current state-of-the-art model in this domain, the COSMIC framework, focuses on injecting the Commonsense Knowledge into a recurrent model encoding the conversation context. 
However, this architecture considers the emotion states as a sequential input that can omit the strong relationships between emotion states, especially in long conversations. Therefore, we propose a new architecture, EdepCOSMIC, to model the dependencies between the emotion states using the self-attention mechanism on top of the COSMIC framework. Experimental results showed that our proposed model worked effectively and got state-of-the-art results on four benchmark datasets in this domain:  IEMOCAP, DailyDialog, EmornyNLP, and MELD.  

### Reference 
This code is based on open-source [conv-emotion](https://github.com/declare-lab/conv-emotion)
### Python Environment 

```bash 
conda env create --prefix env-erc/ python=3.7
conda activate ./env-erc/
pip install -r requirements.txt 
```

### Prepare dataset + preprocessed features 


### Run 

```bash 
conda activate ./env-erc/
cd COSMIC/erc-training/ && bash train_iemocap.sh
```