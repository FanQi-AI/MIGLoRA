# Efficient Multi-Instance Generation with Janus-Pro-Dirven Prompt Parsing
[Arxiv Preprint](https://arxiv.org/abs/2503.21069) | [Huggingface Card](https://huggingface.co/Jonas179/MIGLoRA)

This repo is the code for the paper *Efficient Multi-Instance Generation with Janus-Pro-Dirven Prompt Parsing*

## Main framework:
<img width="1705" height="806" alt="image" src="https://github.com/user-attachments/assets/13a6154e-7ed4-403f-afb9-305b54fec85b" />

## Installation Guide
Before using it, please extract the `diffusers.zip` file to the current directory.

```
git clone https://github.com/FanQi-AI/MIGLoRA.git
cd MIGLoRA
conda create -n miglora python=3.10
conda activate miglora
pip install -r rquirements.txt
cd diffusers
pip install .
cd ..
```
## Inference
Before running the inference script, the model weights need to be downloaded from the [model card](https://huggingface.co/Jonas179/MIGLoRA) on Hugging Face and [stable-diffusion-v1-5 weights](https://huggingface.co/frankjoshua/realisticVisionV51_v51VAE)  to `models`:

```
python inference.py 0 1 1 3.5 200000 "SD15" "UniPCM" "results/inf" "ckpt/checkpoint-unet/pytorch_lora_weights.safetensors" "/ckpt/checkpoint-unet/model_1.safetensors"
```


## Train
Please put the training data in a JSON file with the following format:
```
{
	[
  {
    "image_path": "path/to/image1.jpg",      // Image file path
    "image_width": 800,                      // Original image width (in pixels)
    "image_height": 600,                     // Original image height (in pixels)
    "caption": "A dog playing in the park.", // Full image textual description
    "mask": [                                // List of object masks
      {
        "label": "dog",                      // Object class (skip "background")
        "value": 1,                          // Numeric ID corresponding to the class
        "sub_caption": "a brown dog",        // Localized description of the object (optional, use empty string if none)
        "box": {
          "relative": [                      // Normalized bounding box [x1, y1, x2, y2]
            0.1,
            0.2,
            0.5,
            0.6
          ]
        }
      },
      {
        "label": "ball",
        "value": 2,
        "sub_caption": "",                  // Can be empty string
        "box": {
          "relative": [0.6, 0.3, 0.8, 0.5]
        }
      }
      // 最多可以有任意多个 mask 条目，Dataset 会选取前 max_conditions 个
    ]
  },
  // …更多条目
]
}
```

Before running the training program, please modify the `output_dir` and `data_dir` fields in the configuration file `train_lora.json`, and then execute the script.
```
python train.py
```

## BibTeX
```
@misc{qi2025efficientmultiinstancegenerationjanusprodirven,
      title={Efficient Multi-Instance Generation with Janus-Pro-Dirven Prompt Parsing}, 
      author={Fan Qi and Yu Duan and Changsheng Xu},
      year={2025},
      eprint={2503.21069},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.21069}, 
}
```

