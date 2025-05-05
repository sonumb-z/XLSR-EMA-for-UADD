# Universal Audio Deepfake Detection Using Pretrained XLS-R and Efficient Multi-Scale Attention

This repository contains our implementation of the paper:
 **"Universal Audio Deepfake Detection Using Pretrained XLS-R and Efficient Multi-Scale Attention"**

------

## 📦 Datasets

| Dataset                      | Link                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| ASVspoof 2019                | [Download](https://datashare.is.ed.ac.uk/handle/10283/3336)  |
| Codecfake-mini               | [Download](https://drive.google.com/file/d/19TW1nscU8lCpSkHTA5WvpKto1iIiNZqf/view) |
| ASVspoof 2021 LA             | [Download](https://zenodo.org/record/4837263)                |
| ASVspoof 2021 DF             | [Download](https://zenodo.org/record/4835108)                |
| In-the-Wild                  | [Download](https://deepfake-total.com/in_the_wild)           |
| Codecfake Test (part 1 of 2) | [Download](https://zenodo.org/records/13838823)              |
| Codecfake Test (part 2 of 2) | [Download](https://zenodo.org/records/11125029)              |

| Other Resources | Link                                                         |
| --------------- | ------------------------------------------------------------ |
| keys            | [Download](https://drive.google.com/file/d/1ZRn3s9gJ3os_SC0USh4Zx3xcjwAtu-dQ/view) |
| fairseq         | [Download](https://drive.google.com/file/d/1XitO6TbkWRaYrSEPkuh6Y15-THUNH0TX/view) |
| XLS-R (300M)    | [GitHub](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/xlsr) |
| best_model.pth  | [Download](https://drive.google.com/file/d/1I28mcjMuvY5uWBGVMwyQa_oFOq61yz9i/view) |

------

## 📁 Project Structure

```
XLSR-EMA-for-UADD/
├── core_scripts/
├── database/
├── fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1/
├── keys/
├── logs/
├── models/
├── run_logs/
├── scores/
├── CSAM.py
├── data_utils_SSL.py
├── eval_metric_LA.py
├── eval_metrics_DF.py
├── evaluate_2021_DF.py
├── evaluate_2021_LA.py
├── evaluate_codecfake.py
├── evaluate_in_the_wild.py
├── main.py
├── model.py
├── RawBoost.py
├── README.md
└── requirements.txt
```

------

## ⚙️ Environment Setup

We used **Python 3.7** and **CUDA 11.8** for this project. If you're using a different environment, please install the appropriate PyTorch version from [PyTorch Stable Releases](https://download.pytorch.org/whl/torch_stable.html).

```
$ git clone https://github.com/sonumb-z/XLSR-EMA-for-UADD

$ cd XLSR-EMA-for-UADD
$ unzip fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1.zip
$ unzip keys.zip

$ conda create -n XLSR-EMA python=3.7
$ conda activate XLSR-EMA
$ pip install torch==1.8.1+cu111 torchaudio==0.8.1
$ cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
$ pip install --editable ./
$ cd ..
$ pip install -r requirements.txt
```

------

## 🏋️ Training

To train the model, run:

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --lr=0.000001 \
    --batch_size=10 \
    --loss=WCE \
    --seed=222 \
    --weight_decay=0 \
    --algo=333 \
    --train_task=co-train \
    --model=ModelEMA \
    --num_epochs=100 \
    --CSAM
```

------

## 🔍 Testing

### Evaluate on All Datasets

To evaluate the pre-trained model on **ASVspoof 2021 DF**, **ASVspoof 2021 LA**, **In-the-Wild**, and **Codecfake**, modify the dataset paths accordingly and run:

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --track=all \
    --model=ModelEMA \
    --tag=best \
    --is_eval \
    --eval \
    --model_path=./models/best_model.pth
```

> This will generate 11 `score.txt` files (including 7 for Codecfake). These files are used to calculate the Equal Error Rate (EER %).
>  Estimated time on RTX 4090: ~9 hours.

------

### Compute EER (%)

Run the following scripts to compute EER:

```
echo "in_the_wild"
python evaluate_in_the_wild.py ./scores/scores_In-the-Wild_best.txt ./keys eval

echo "21LA"
python evaluate_2021_LA.py ./scores/scores_21LA_best.txt /lab/songziwen/data/keys/ eval

echo "21DF"
python evaluate_2021_DF.py ./scores/scores_21DF_best.txt /lab/songziwen/data/keys/DF eval

echo "codecfake"
python evaluate_codecfake.py /lab/songziwen/data/Codecfake/label/ best
```

------

### Evaluate Individual Datasets

#### ASVspoof 2021 LA:

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --track=LA \
    --is_eval \
    --eval \
    --model_path=models/best_model.pth \
    --protocols_path=./database/ASVspoof_DF_cm_protocols/ASVspoof2021.LA.cm.eval.trl.txt \
    --database_path=/lab/songziwen/data/LA/ASVspoof2021_LA_eval/ \
    --eval_output=./scores/scores_21LA_best.txt
```

#### ASVspoof 2021 DF:

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --track=DF \
    --is_eval \
    --eval \
    --model_path=models/best_model.pth \
    --protocols_path=./database/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt \
    --database_path=/lab/songziwen/data/DF/ASVspoof2021_DF_eval/ \
    --eval_output=./scores/scores_21DF_best.txt
```

#### In-the-Wild:

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --track=In-the-Wild \
    --model=ModelEMA \
    --is_eval \
    --eval \
    --model_path=./models/best_model.pth \
    --protocols_path=database/ASVspoof_DF_cm_protocols/in_the_wild.eval.txt \
    --database_path=/lab/songziwen/data/InTheWild/release_in_the_wild/ \
    --eval_output=./scores/scores_In-the-Wild_best.txt
```

#### Codecfake:

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --track=codecfake \
    --is_eval \
    --eval \
    --database_path=/lab/songziwen/data/Codecfake/ \
    --model_path=./models/best_model.pth \
    --eval_output=./scores/ \
    --tag=best
```

------

## 📊 Results Using Pre-Trained Model XLS-R AND EMA

| Dataset          | EER (%) |
| ---------------- | ------- |
| ASVspoof 2021 LA | 3.44%   |
| ASVspoof 2021 DF | 1.24%   |
| In-the-Wild      | 5.25%   |
| Codecfake        | 4.64%   |

## 🙏 Acknowledgements

We would like to express our gratitude to the following open-source projects for providing valuable code references and inspiration:

- [SLSforASVspoof-2021-DF](https://github.com/QiShanZhang/SLSforASVspoof-2021-DF)
- [Codecfake](https://github.com/xieyuankun/Codecfake)
- [SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)
