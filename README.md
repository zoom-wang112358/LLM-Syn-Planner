# LLM-Syn-Planner

[ICML 2025] [LLM-Augmented Chemical Synthesis and Design Decision Programs](https://openreview.net/forum?id=NhkNX8jYld)


## About

LLM-Syn-Planner is an LLM-based retrosynthesis planning framework.


## Setups
You need to get an API key for GPT-4o and DeepSeek-V3.

### Package Installation
```bash
conda create -n syn python=3.9
conda activate syn
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install PyTDC 
pip install PyYAML
pip install rdkit
pip install "syntheseus[all]"
pip install selfies
pip install rdchiral 
```

Then we can activate conda via following command. 
```bash
conda activate syn 
```


### Experiments

Download and unzip the files from this [link](https://www.dropbox.com/scl/fi/dmmypid2ooohp3freiox8/dataset.zip?rlkey=fmrhvds6fmxck2cp8h94albpc&st=8fmtxls4&dl=0), and put dataset/ under the current directory. To run experiments:

```bash
# USPTO easy
python main.py --method planning --dataset_name USPTO-easy --max_oracle_calls 100
# USPTO 190
python main.py --method planning --dataset_name USPTO-190 --max_oracle_calls 100
# Pistachio reachable
python main.py --method planning --dataset_name pistachio_reachable --max_oracle_calls 100
# Pistachio hard
python main.py --method planning --dataset_name pistachio_hard --max_oracle_calls 100
```

In the experiments, we use similarity search based on [retrosim](https://github.com/connorcoley/retrosim). We download datasets from [Retro*](https://github.com/binghong-ml/retro_star) and [DESP](https://github.com/coleygroup/desp).

## Citation
If you find our work helpful, please consider citing our paper:

```
@article{wang2025llm,
  title={LLM-Augmented Chemical Synthesis and Design Decision Programs},
  author={Haorui Wang and Jeff Guo and Lingkai Kong and Rampi Ramprasad and Philippe Schwaller and Yuanqi Du and Chao Zhang},
  journal={Forty-Second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=NhkNX8jYld}
}

```
