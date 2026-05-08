# Custom Image Classifier

A modern desktop GUI application built with **Python, Tkinter, PyTorch, and Torchvision** that allows you to create your own custom image classes, train a classifier, and classify single images or entire folders.

This project uses a **pretrained ResNet-18 backbone** as a feature extractor and trains a lightweight custom classification head on your own image categories.

---

## ✨ Features

- 🖼️ Classify a **single image**
- 📁 Batch classify an **entire folder**
- 🧠 Uses pretrained **ResNet-18** transfer learning
- 🏷️ Add and remove custom classes
- 🎯 Train on your own image datasets
- 📊 Training progress with loss logs
- 📈 Folder classification progress + ETA
- 👀 Live image preview
- 🌙 Modern styled Tkinter UI
- 🖼️ Supports multiple image formats
 
---

# 📂 Project Structure

```bash
project/
│── main.py
│── custom_head.pth          # Saved trained classifier weights
│── custom_classes.json      # Saved class labels
│── images/                  # Default folder for batch classification
│── custom_classes/
│    ├── cats/
│    ├── dogs/
│    └── cars/
