# IRE Project: Abstractive Text Summarisation

1. MCLAS\
Code: [Link to mT5 repo]
[Paper](https://doi.org/10.48550/arXiv.2105.13648)

2. mT5\
Code: [Link to repo]

Training vanilla mT5. Currently it is in Gujarati: [Training of mT5](https://colab.research.google.com/drive/1TDdo58cIKl4vrDjhgtX5CBZ5lbFPU9jd?usp=sharing) (The Colab files have the log data).

Predicting summary by mT5.
[Predicting mT5](https://colab.research.google.com/drive/1BLPfMinTlLz9ttwKV4VHKNCVk069T0iZ?usp=sharing)

[This](https://drive.google.com/drive/folders/13HXeMVUhky1nJxsGO-W2eI-cEPHnsYe3?usp=sharing) contains fine-tuned Gujarati Summarisation model (trained up to 3 epochs).

[Paper](https://doi.org/10.48550/arXiv.2010.11934)

Fine-tuned code for mT5. [Fine-tuned mT5](https://colab.research.google.com/drive/1zmuhDapQPA1g_Uswim4b1gxNFjBaQBde?usp=sharing)
### Datasets used:
[Gujarati](https://drive.google.com/file/d/1hiHwpTNMG-jcj3n-wDgzjJ2tOptcWfAL/view?usp=sharing)

[Hindi and English](https://drive.google.com/file/d/1PMquHwtPC_lbeorJgXd7fJsCT8008QhC/view?usp=sharing), ignore or replace gujarati folder in this.

3. IndicBART
[Code](https://colab.research.google.com/drive/1UqkmIYp0VD9HGavWkh3cDy4sH15plEyj?usp=sharing)


## Results:

Vanilla mT5:

|Language|Rouge-1|Rouge-2|Rouge-L|
|---|---|---|---|
|English| 48.6645| 36.1859|43.69|
|Hindi| 51.468| 40.1589| 46.6524|
|Gujarati|  23.0882|14.0092|20.7578|

XL-Sum:
|Language|Rouge-1|Rouge-L|
|---|---|---|
|English|43.35|34.47|
|Hindi|41.79|36.67|


Contributors:\
Aaradhya Gupta\
Devesh Marwah\
Mayank Goel\
Radheshyam Thiyagarajan


