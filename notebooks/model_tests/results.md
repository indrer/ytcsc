## Accuracy table of the best performing models

* WV - Pretrained word vectors
* CV - Cross-validation
* EV - Evaluation
* M1 - Basic model (Single LSTM, BISTM, CNN-LSTM or CNN-BILSTM layers)
* M2 - Basic model with additional layers

|   | CV  | EV  | CV + WV  | EV + WV  | IMDB + CV  | IMDB + EV   |
|---|---|---|---|---|---|---|
| LSTM M1| 80.35%  | 81.16%  | 81.32%  | 82.13%  | 70.22%  | 87.4%  |
| LSTM M2| 81.28%  | 80.80%  | 82.97%  | 82.37%  | 77.63%  | 87.46%  |
| BILSTM M1 | **83.45%**  | 81.88%  | **83.82%**  | 82%  | 79.45%  | 87.33%  |
| BILSTM M2 | 83.21%  | 81.4%  | 80.56%  | 79.47%  | 79.99%  | 87.21%  |
| CNN-LSTM M1 | 81.88%  | 82%  | 79.47%  | 80.92%  | 74.39%  | 87.42%  |
| CNN-LSTM M2 | 79.35%  | 81.64%  | 79.35%  | 82.13%  | 79.87%  | 86.57%  |
| CNN-BILSTM M1 | 82.61%  | 83.7%  | 80.07%  | 80.07%  | 72.95%  | 86.75%  |
| CNN-BILSTM M1 | 82.97%  | 81.64%  | 80.07%  | 78.38%  | 74.32%  | 87.38%  |