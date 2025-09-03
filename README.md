
# Food Vision 101 - Model Deployment

A computer vision project that classifies food images using PyTorch with EfficientNetB2 and Vision Transformer (ViT) models, featuring deployment through Gradio web interface.

## 📋 Project Overview

This project implements **FoodVision Mini** and **FoodVision Big** - food classification systems that can identify different types of food from images. The project focuses on achieving both high performance (95%+ accuracy) and fast inference speed (~30 FPS for real-time applications).

### Models Implemented:
- **FoodVision Mini**: Classifies 3 food classes (pizza, steak, sushi)
- https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip

- **FoodVision Big**: Classifies 101 food classes from the Food101 dataset

### Key Features:
- ✅ High-performance food classification
- ⚡ Real-time inference optimization
- 🚀 Interactive Gradio web demo
- 📊 Comprehensive model comparison and evaluation
- 🌐 Deployable to Hugging Face Spaces

## 🏗️ Architecture

### Models Compared:
1. **EfficientNetB2 Feature Extractor**
   - Pre-trained on ImageNet
   - Modified classifier head for food classes
   - Optimized for speed and efficiency

2. **Vision Transformer (ViT-B/16)**
   - Pre-trained ViT with 16x16 patches
   - Custom head for classification
   - Higher accuracy but slower inference

## 🎯 Performance Goals

- **Accuracy**: 95%+ on test dataset
- **Speed**: ~30 FPS (0.03 seconds per inference)
- **Deployment**: Real-time web application

## 📁 Project Structure

```
Food_Vision_Deployment.ipynb     # Main notebook
├── going_modular/               # Modular PyTorch utilities
│   ├── data_setup.py           # Data loading utilities
│   ├── engine.py               # Training engine
│   └── utils.py                # General utilities
├── models/                     # Saved model weights
├── demos/foodvision_mini/      # Deployment files
│   ├── app.py                  # Gradio application
│   ├── model.py                # Model definitions
│   ├── requirements.txt        # Dependencies
│   └── examples/               # Demo images
└── helper_functions.py         # Helper utilities
```

## 🚀 Getting Started

### Prerequisites

```bash
torch>=1.12
torchvision>=0.13
gradio
matplotlib
torchinfo
pandas
Pillow
```

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NANDAGOPALNG/Food_Vision_101.git
   cd Food_Vision_101
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the going_modular utilities** (auto-downloaded in notebook):
   ```bash
   git clone https://github.com/NANDAGOPALNG/pytorch--going_modular
   ```

## 🔧 Usage

### 1. Training Models

The notebook includes complete training pipelines for both EfficientNetB2 and ViT models:

```python
# Create EfficientNetB2 model
effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=3)

# Train the model
effnetb2_results = engine.train(
    model=effnetb2,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    epochs=10,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device
)
```

### 2. Model Evaluation

Compare models across multiple metrics:
- Test accuracy and loss
- Inference time per prediction
- Model size and parameters
- Speed vs. performance tradeoffs

### 3. Making Predictions

```python
# Load and predict on single image
pred_labels_and_probs, pred_time = predict(image)
```

### 4. Gradio Demo

Launch interactive web interface:

```python
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(num_top_classes=3), gr.Number()],
    title="FoodVision Mini 🍕🥩🍣"
)
demo.launch()
```

## 📊 Model Performance Comparison

| Model | Test Accuracy | Inference Time (CPU) | Model Size | Parameters |
|-------|---------------|----------------------|------------|------------|
| EfficientNetB2 | ~95%+ | ~0.03s | ~28MB | ~7.7M |
| ViT-B/16 | ~96%+ | ~0.08s | ~80MB+ | ~85M+ |

**Winner for Deployment**: EfficientNetB2 (better speed-performance tradeoff)

## 🌐 Deployment

### Local Deployment

1. **Navigate to demo folder**:
   ```bash
   cd demos/foodvision_mini
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   python app.py
   ```

### Hugging Face Spaces Deployment

1. Create new Space on [Hugging Face](https://huggingface.co/new-space)
2. Clone the repository locally
3. Copy demo files to the cloned repo
4. Install Git LFS for large model files:
   ```bash
   git lfs install
   git lfs track "*.pth"
   ```
5. Push to Hugging Face:
   ```bash
   git add .
   git commit -m "Deploy FoodVision Mini"
   git push
   ```

## 🍕 FoodVision Mini vs Big

### FoodVision Mini
- **Classes**: 3 (pizza, steak, sushi)
- **Dataset**: 20% of Food101 subset
- **Focus**: Speed and deployment optimization
- **Use Case**: Fast food classification demo

### FoodVision Big
- **Classes**: 101 (full Food101 dataset)
- **Dataset**: Complete Food101 (101 classes × 1000 images each)
- **Focus**: Comprehensive food recognition
- **Use Case**: Production food classification system

## 📈 Key Insights

1. **EfficientNetB2** offers the best speed-accuracy tradeoff for deployment
2. **Data augmentation** with TrivialAugmentWide improves generalization
3. **CPU optimization** is crucial for real-world deployment
4. **Gradio** provides an excellent framework for ML demo deployment

## 🛠️ Technical Implementation

### Model Creation Functions
- `create_effnetb2_model()`: Creates EfficientNetB2 feature extractor
- `create_vit_model()`: Creates Vision Transformer feature extractor
- `pred_and_store()`: Batch prediction and timing utility
- `predict()`: Single image prediction for Gradio

### Data Processing
- Custom transforms for training (with augmentation) and testing
- Food101 dataset integration with PyTorch
- Dynamic dataset splitting utilities

### Deployment Pipeline
- Model serialization and loading
- Gradio interface creation
- Hugging Face Spaces integration
- Requirements management

## 📝 Results and Findings

The project successfully demonstrates:
- ✅ Both models achieve >95% accuracy on food classification
- ⚡ EfficientNetB2 provides ~3x faster inference than ViT
- 🎯 Real-time food classification is achievable with proper model selection
- 🌐 Seamless deployment to web interfaces through Gradio

## 🔮 Future Improvements

- [ ] Implement model quantization for faster inference
- [ ] Add support for mobile deployment (ONNX/CoreML)
- [ ] Expand to more food categories
- [ ] Implement confidence thresholding
- [ ] Add batch processing capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Learn PyTorch](https://www.learnpytorch.io/) for comprehensive tutorials
- [Hugging Face](https://huggingface.co/) for deployment platform
- [Food101 Dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.Food101.html) creators
- PyTorch and torchvision teams

## 📬 Contact

**NANDAGOPAL NG** - [GitHub Profile](https://github.com/NANDAGOPALNG)

Project Link: [https://github.com/NANDAGOPALNG/Food_Vision_101](https://github.com/NANDAGOPALNG/Food_Vision_101)

---

*Built with ❤️ using PyTorch, Gradio, and modern MLOps practices*
