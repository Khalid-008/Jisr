# Next Steps — Arabic Sign Language Translation System

## Step 1: Set Up Gemini API Key

1. Go to https://aistudio.google.com/apikey
2. Click "Create API Key"
3. Copy the key
4. In the project folder, create a file called `.env` with this content:
   ```
   GEMINI_API_KEY=your_key_here
   ```

---

## Step 2: Download the Arabic Sign Language Dataset

1. Install Kaggle CLI:
   ```
   pip install kaggle
   ```

2. Get your Kaggle API key:
   - Go to https://www.kaggle.com/settings
   - Scroll to "API" section
   - Click "Create New Token"
   - This downloads a file called `kaggle.json`

3. Place `kaggle.json` in:
   - **Windows**: `C:\Users\<your_username>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`

4. Download the dataset:
   ```
   kaggle datasets download -d ammarsayedtaha/arabic-sign-language-dataset-2022
   ```

5. Extract the downloaded zip file into `data/raw/` folder

---

## Step 3: Prepare the Dataset

Run:
```
python prepare_dataset.py
```

This will:
- Organize images into `data/images/train/` and `data/images/val/`
- Create matching labels in `data/labels/train/` and `data/labels/val/`
- Generate `data.yaml` for YOLOv5 training

---

## Step 4: Train the Model (Google Colab)

1. Go to https://colab.research.google.com
2. Click **File > Upload notebook**
3. Upload `train_colab.ipynb` from this project
4. In Colab, go to **Runtime > Change runtime type > GPU (T4)**
5. Run all cells in order:
   - It will ask you to upload your `kaggle.json` file
   - Training takes ~1-2 hours on a free GPU
   - The trained model will be saved to your Google Drive
6. Download `arabic_sign_best.pt` from Google Drive
7. Place it in the `models/` folder of this project

---

## Step 5: Run the Application

```
python app.py
```

### How to Use

1. Click **Start** to turn on the webcam
2. Show Arabic sign language gestures to the camera
3. The detected Arabic letters/words appear in the Arabic text box
4. Click **Translate** to translate to English
5. Toggle between **Text** mode (English text) and **Sign (ASL)** mode (ASL fingerspelling animation)
6. Click **Clear** to reset everything

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Model Not Found" error | Make sure `models/arabic_sign_best.pt` exists — complete Step 4 |
| Translation says "check GEMINI_API_KEY" | Create `.env` file with your Gemini API key — see Step 1 |
| Webcam not opening | Check if another app is using the camera, or try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `detector.py` |
| Low detection accuracy | Try better lighting, plain background, and clear hand gestures |
| Kaggle download fails | Make sure `kaggle.json` is in the correct folder with correct permissions |
