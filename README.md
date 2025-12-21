# Audio Deepfake Detection Using Pretrained XLS-R and Efficient Multi-Scale Attention

This repository contains our implementation of the paper:
 **"Audio Deepfake Detection Using Pretrained XLS-R and Efficient Multi-Scale Attention"**

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

The project was developed and tested using the following hardware and software configuration:

- **GPU**: NVIDIA RTX 4090
- **Python**: 3.7
- **CUDA**: 11.8

**Experimental Details & Reproducibility:**

Audio data are cropped or concatenated giving segments of approximately 4 seconds duration (64,600 samples). The Adam optimizer was configured with a learning rate of $10^{-6}$, a weight decay coefficient of 0, and a batch size of 10. Training proceeded for 100 epochs with an early stopping criterion (patience = 3). The model was trained on a single NVIDIA GeForce RTX 4090 GPU, and experimental results can be reproduced using the same random seed and GPU environment.

If you're using a different environment, please install the appropriate PyTorch version from [PyTorch Stable Releases](https://download.pytorch.org/whl/torch_stable.html).

```bash
$ git clone [https://github.com/sonumb-z/XLSR-EMA-for-UADD](https://github.com/sonumb-z/XLSR-EMA-for-UADD)

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

```bash
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

```bash
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

```bash
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

```bash
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

```bash
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

```bash
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

```bash
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

## 📊 Experimental Results & Analysis

### Comparison with SOTA Methods

**Table 1: Performance on In-The-Wild and Codecfake evaluation sets.** Bold indicates the best EER(%) performance.

| System | In-The-Wild EER(%) | Codecfake EER(%) |
| :--- | :---: | :---: |
| RawNet2 [24] | 33.94 | 50.22 |
| XLSR+AASIST [17] | 10.46 | - |
| XLSR+TCM [25] | 7.79 | 35.72 |
| XLSR+SLS [1] | 7.46 | 33.43 |
| XLSR+Mamba [26] | 6.71 | 35.26 |
| **Ours (XLSR+EMA)** | 7.90 | 28.78 |
| **Ours + Add. Data** | **5.25** | **4.64** |

**Table 2: Performance comparison with SOTA single systems on the ASVspoof 2021 LA and DF evaluation sets**.

| System | 2021 LA EER(%) | 2021 LA min t-DCF | 2021 DF EER(%) |
| :--- | :---: | :---: | :---: |
| RawNet2 [24] | 5.31 | 0.310 | 22.38 |
| SE-Rawformer [27] | 4.98 | 0.318 | 20.26 |
| XLSR+ASSIST [17] | **0.82** | **0.206** | 2.85 |
| WavLM+MFA [28] | 5.08 | - | 2.56 |
| XLSR+SLS [1] | 2.87 | - | 1.92 |
| XLSR+Mamba [26] | 0.93 | 0.208 | 1.88 |
| **Ours (XLSR+EMA)** | 1.59 | 0.230 | 2.39 |
| **Ours + Add. Data** | 3.44 | 0.270 | **1.24** |

### Ablation Studies

<p><strong>Table 3: Ablation study results across various configurations and datasets.</strong> The evaluation metric is EER(%). <em>g</em>: Number of EMA groups. <em>ITW</em>: In-The-Wild. <em>Codec⁺</em>: Codecfake Test.</p>

<table>
  <thead>
    <tr>
      <th align="left">Ablation</th>
      <th align="center">Cfg.</th>
      <th align="center">21LA</th>
      <th align="center">21DF</th>
      <th align="center">ITW</th>
      <th align="center">Codec⁺</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" align="left">ours</td>
      <td align="center">g=1</td>
      <td align="center"><strong>3.44</strong></td>
      <td align="center"><strong>1.24</strong></td>
      <td align="center"><strong>5.25</strong></td>
      <td align="center">4.64</td>
    </tr>
    <tr>
      <td align="center">g=2</td>
      <td align="center">3.69</td>
      <td align="center">1.28</td>
      <td align="center">6.55</td>
      <td align="center"><strong>3.81</strong></td>
    </tr>
    <tr>
      <td rowspan="2" align="left">w/o EMA</td>
      <td align="center">SLS</td>
      <td align="center">4.77</td>
      <td align="center">1.31</td>
      <td align="center">6.25</td>
      <td align="center">4.02</td>
    </tr>
    <tr>
      <td align="center">-</td>
      <td align="center">3.56</td>
      <td align="center">1.32</td>
      <td align="center">6.73</td>
      <td align="center">3.81</td>
    </tr>
    <tr>
      <td align="left">w/o DA</td>
      <td align="center">g=1</td>
      <td align="center">5.91</td>
      <td align="center">2.23</td>
      <td align="center">6.61</td>
      <td align="center">3.99</td>
    </tr>
  </tbody>
</table>


### Robustness Analysis (Data Augmentation Comparison)

We compared our proposed method with different RawBoost data augmentation techniques. The methods are defined as follows:
* **Method 1**: Convolutive noise
* **Method 2**: Impulsive noise
* **Method 3**: Coloured additive noise

<p><strong>Table 4: Performance comparison (EER%) with different data augmentation methods across datasets.</strong> Bold indicates the best performance.</p>

<table>
  <thead>
    <tr>
      <th rowspan="2" align="center">Method</th>
      <th rowspan="2" align="center">21LA</th>
      <th rowspan="2" align="center">21DF</th>
      <th rowspan="2" align="center">In the Wild</th>
      <th colspan="8" align="center">codecfake</th>
    </tr>
    <tr>
      <th align="center">C1</th>
      <th align="center">C2</th>
      <th align="center">C3</th>
      <th align="center">C4</th>
      <th align="center">C5</th>
      <th align="center">C6</th>
      <th align="center">C7</th>
      <th align="center">avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">1</td>
      <td align="center">5.14/0.3014</td>
      <td align="center">2.49</td>
      <td align="center">6.18</td>
      <td align="center">0.08</td>
      <td align="center"><strong>0.82</strong></td>
      <td align="center">0.46</td>
      <td align="center">0.45</td>
      <td align="center">0.24</td>
      <td align="center"><strong>2.14</strong></td>
      <td align="center"><strong>17.96</strong></td>
      <td align="center"><strong>3.16</strong></td>
    </tr>
    <tr>
      <td align="center">2</td>
      <td align="center">4.15/0.2836</td>
      <td align="center">1.40</td>
      <td align="center">6.66</td>
      <td align="center"><strong>0.07</strong></td>
      <td align="center">1.30</td>
      <td align="center"><strong>0.36</strong></td>
      <td align="center">0.98</td>
      <td align="center">0.18</td>
      <td align="center">2.75</td>
      <td align="center">26.66</td>
      <td align="center">4.62</td>
    </tr>
    <tr>
      <td align="center">3</td>
      <td align="center">4.26/0.2773</td>
      <td align="center">1.35</td>
      <td align="center">6.37</td>
      <td align="center">0.09</td>
      <td align="center">1.31</td>
      <td align="center">0.62</td>
      <td align="center">0.48</td>
      <td align="center">0.26</td>
      <td align="center">3.01</td>
      <td align="center">20.89</td>
      <td align="center">3.81</td>
    </tr>
    <tr>
      <td align="center">Ours</td>
      <td align="center"><strong>3.44/0.270</strong></td>
      <td align="center"><strong>1.24</strong></td>
      <td align="center"><strong>5.25</strong></td>
      <td align="center">0.08</td>
      <td align="center">1.12</td>
      <td align="center">0.45</td>
      <td align="center"><strong>0.42</strong></td>
      <td align="center"><strong>0.17</strong></td>
      <td align="center">3.52</td>
      <td align="center">26.71</td>
      <td align="center">4.64</td>
    </tr>
  </tbody>
</table>


<p><strong>Table 5: Performance comparison (EER%) across different SNR ranges using our proposed data augmentation.</strong> Bold indicates the best performance.</p>

<table> <thead> <tr> <th rowspan="2" align="center">SNR</th> <th rowspan="2" align="center">21LA</th> <th rowspan="2" align="center">21DF</th> <th rowspan="2" align="center">In the Wild</th> <th colspan="8" align="center">codecfake</th> </tr> <tr> <th align="center">C1</th> <th align="center">C2</th> <th align="center">C3</th> <th align="center">C4</th> <th align="center">C5</th> <th align="center">C6</th> <th align="center">C7</th> <th align="center">avg</th> </tr> </thead> <tbody> <tr> <td align="center">[10,40]</td> <td align="center">4.64/0.2900</td> <td align="center">1.47</td> <td align="center">7.52</td> <td align="center">0.08</td> <td align="center">1.35</td> <td align="center"><strong>0.29</strong></td> <td align="center">0.72</td> <td align="center">0.26</td> <td align="center"><strong>1.91</strong></td> <td align="center">28.64</td> <td align="center">4.75</td> </tr> <tr> <td align="center">[10,30]</td> <td align="center">3.85/0.2721</td> <td align="center"><strong>1.12</strong></td> <td align="center">6.86</td> <td align="center">0.08</td> <td align="center">2.06</td> <td align="center">0.79</td> <td align="center">0.51</td> <td align="center">0.23</td> <td align="center">4.81</td> <td align="center"><strong>24.28</strong></td> <td align="center">4.68</td> </tr> <tr> <td align="center">[0,30]</td> <td align="center"><strong>3.44/0.270</strong></td> <td align="center">1.24</td> <td align="center"><strong>5.25</strong></td> <td align="center"><strong>0.08</strong></td> <td align="center"><strong>1.12</strong></td> <td align="center">0.45</td> <td align="center"><strong>0.42</strong></td> <td align="center"><strong>0.17</strong></td> <td align="center">3.52</td> <td align="center">26.71</td> <td align="center"><strong>4.64</strong></td> </tr> </tbody> </table>

## 🙏 Acknowledgements

We would like to express our gratitude to the following open-source projects for providing valuable code references and inspiration:

- [SLSforASVspoof-2021-DF](https://github.com/QiShanZhang/SLSforASVspoof-2021-DF)
- [Codecfake](https://github.com/xieyuankun/Codecfake)
- [SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)