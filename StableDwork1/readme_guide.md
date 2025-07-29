# Stable Diffusion CLI Training & Generation System

A complete command-line solution for training custom Stable Diffusion models and generating images locally on macOS, with no cloud dependencies.

## Features

- 🏋️ **Custom Model Training**: Train SD models on your own datasets
- 🎨 **Local Image Generation**: Generate images entirely on your machine
- 📦 **Easy Setup**: One-command installation and dependency management
- 🔄 **Auto-Updates**: Keep your system up to date
- 💾 **Dataset Management**: Organize and prepare training data
- ⚡ **MPS Acceleration**: Uses Apple Silicon GPU acceleration when available
- 🔧 **CLI Interface**: Simple, scriptable command-line interface

## System Requirements

- macOS 10.15+ (Catalina or later)
- Python 3.8+
- 8GB+ RAM (16GB+ recommended for training)
- 10GB+ free disk space
- Apple Silicon Mac recommended for best performance

## Installation

### Quick Install

```bash
curl -sSL https://raw.githubusercontent.com/your-repo/sd-cli/main/install.sh | bash
```

### Manual Install

1. Download the installation script:
```bash
wget https://raw.githubusercontent.com/your-repo/sd-cli/main/install.sh
chmod +x install.sh
./install.sh
```

2. Run initial setup:
```bash
sd setup
```

This will:
- Create a virtual environment
- Install all required dependencies (PyTorch, Diffusers, etc.)
- Download the base Stable Diffusion v1.5 model
- Set up directory structure

## Quick Start

### 1. Prepare Your Dataset

Organize your training images in a folder with optional caption files:

```
my_images/
├── image1.jpg
├── image1.txt  # "a beautiful landscape"
├── image2.png
├── image2.txt  # "portrait of a person"
└── ...
```

Prepare the dataset:
```bash
sd dataset prepare --path ./my_images --name my_custom_dataset
```

### 2. Train a Custom Model

```bash
sd train --dataset my_custom_dataset --model my_custom_model --epochs 100
```

Training options:
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Training batch size (default: 1)
- `--learning-rate`: Learning rate (default: 1e-6)

### 3. Generate Images

```bash
sd generate --prompt "a beautiful sunset over mountains" --model my_custom_model
```

Generation options:
- `--num-images`: Number of images to generate (default: 1)
- `--steps`: Inference steps (default: 50)
- `--guidance`: Guidance scale (default: 7.5)
- `--model`: Specific model to use (omit for base model)

## Commands Reference

### Setup & Management

```bash
# Initial setup
sd setup

# Update system
sd update

# List available models
sd list models

# List available datasets
sd list datasets
```

### Dataset Management

```bash
# Prepare a new dataset
sd dataset prepare --path /path/to/images --name dataset_name

# List all datasets
sd dataset list
```

### Model Training

```bash
# Basic training
sd train --dataset my_dataset --model my_model

# Advanced training with custom parameters
sd train \
  --dataset my_dataset \
  --model my_model \
  --epochs 200 \
  --batch-size 2 \
  --learning-rate 5e-6
```

### Image Generation

```bash
# Basic generation
sd generate --prompt "your prompt here"

# Generation with custom model
sd generate --prompt "your prompt here" --model my_custom_model

# Batch generation
sd generate \
  --prompt "multiple variations of the same concept" \
  --model my_model \
  --num-images 4 \
  --steps 30 \
  --guidance 8.0
```

## Directory Structure

The system creates the following structure in `~/.sd_manager/`:

```
~/.sd_manager/
├── models/          # Trained models
│   ├── stable-diffusion-v1-5/  # Base model
│   └── my_custom_model/         # Your trained models
├── datasets/        # Prepared datasets
│   └── my_dataset/
│       ├── images/
│       └── metadata.jsonl
├── outputs/         # Generated images
├── venv/           # Python virtual environment
└── config.json    # System configuration
```

## Training Tips

### Dataset Preparation

1. **Image Quality**: Use high-quality images (512x512 or higher)
2. **Consistency**: Keep similar style/subject matter for best results
3. **Captions**: Provide descriptive captions in .txt files
4. **Quantity**: 50-500 images typically work well for fine-tuning

### Training Parameters

- **Small datasets (50-100 images)**: Use higher learning rates (1e-5)
- **Large datasets (500+ images)**: Use lower learning rates (1e-6)
- **Style training**: Lower epochs (50-100)
- **Subject training**: Higher epochs (100-300)

### Hardware Optimization

- **Apple Silicon**: Automatically uses MPS acceleration
- **Intel Macs**: Falls back to CPU (slower but functional)
- **Memory**: Reduce batch size if you encounter OOM errors

## Troubleshooting

### Common Issues

**"ModuleNotFoundError"**
```bash
sd setup  # Reinstall dependencies
```

**"Out of memory" during training**
- Reduce `--batch-size` to 1
- Lower image resolution in training config
- Close other applications

**Slow generation on Intel Macs**
- This is expected - consider using lower `--steps` (20-30)
- Use smaller models when available

**Training takes too long**
- Reduce `--epochs`
- Increase `--learning-rate` slightly
- Use smaller dataset

### Getting Help

```bash
sd --help                    # General help
sd train --help             # Training options
sd generate --help          # Generation options
```

## Advanced Usage

### Custom Training Scripts

For advanced users, you can modify the training configuration by editing the generated training scripts in `~/.sd_manager/`.

### Integration with Other Tools

The CLI can be integrated into workflows:

```bash
# Batch process multiple prompts
echo "prompt 1\nprompt 2\nprompt 3" | while read prompt; do
  sd generate --prompt "$prompt" --model my_model
done
```

### Experiment Tracking

Training automatically logs to the console. For advanced tracking, the system supports Weights & Biases integration (configure in the training script).

## Performance Benchmarks

**Apple M1 Pro (16GB RAM)**:
- Training: ~2-3 min/epoch for 100 images
- Generation: ~30-60 seconds per image

**Apple M2 Max (32GB RAM)**:
- Training: ~1-2 min/epoch for 100 images  
- Generation: ~15-30 seconds per image

**Intel Mac (16GB RAM)**:
- Training: ~10-15 min/epoch for 100 images
- Generation: ~2-5 minutes per image

## Contributing

This is an open-source project. Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Changelog

### v1.0.0
- Initial release
- Basic training and generation
- macOS MPS support
- CLI interface

### Roadmap

- [ ] LoRA training support
- [ ] Multi-GPU training
- [ ] Custom schedulers
- [ ] Image-to-image generation
- [ ] Inpainting support
- [ ] Model merging tools
- [ ] Web UI interface
