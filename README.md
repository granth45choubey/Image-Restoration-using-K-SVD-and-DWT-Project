# Image-Restoration-using-K-SVD-and-DWT-Project
Interactive image restoration tool that adds adjustable noise to input images and denoises them using K-SVD and DWT. Features real-time sliders for noise and denoising control, enabling visual comparison of both techniques for better understanding and performance analysis.

#  Image Restoration using K-SVD and DWT

An interactive image restoration system that demonstrates how noisy images can be effectively denoised using **K-SVD (sparse representation)** and **Discrete Wavelet Transform (DWT)**.

---

##  Overview

This project allows users to:

* Add controlled noise to an input image using a **slider**
* Apply two denoising techniques: **K-SVD** and **DWT**
* Adjust denoising strength dynamically
* **Visually compare** the performance of both methods

It provides a hands-on understanding of modern image restoration techniques.

---

## ✨ Features

*  Adjustable **Noise Slider**
*  K-SVD based sparse coding denoising
*  DWT-based multi-resolution denoising
*  Side-by-side comparison of outputs
*  Denoising strength control for better tuning
*  (Optional) Metrics support like PSNR / SSIM

---

## 🛠️ Tech Stack

* **Python**
* NumPy
* OpenCV
* PyWavelets (for DWT)
* Matplotlib / UI tools

---

## 📂 Project Structure

```
.
├── app.py              # Main application
├── ksvd.py             # K-SVD implementation
├── wavelet.py          # DWT implementation
├── utils.py            # Helper functions
├── metrics.py          # Evaluation metrics
├── requirements.txt    # Dependencies
```

---

## ⚙️ How to Run

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/Image-Restoration-using-K-SVD-and-DWT-Project.git
cd Image-Restoration-using-K-SVD-and-DWT-Project
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the application

```
python app.py
```

---

##  Demo

>  Add screenshots or a GIF here showing:
>
> * Noise slider
> * K-SVD output
> * DWT output
> * Comparison view

---

##  Results & Analysis

* K-SVD provides better **detail preservation**
* DWT offers faster **multi-scale denoising**
* Users can visually and quantitatively compare results

---

##  Use Cases

* Image preprocessing
* Computer vision pipelines
* Educational tool for signal/image processing
* Understanding sparse representation techniques

---
## ⚙️ Pipeline

```text
Original Image
     ↓
Add Gaussian Noise (σ)
     ↓
Apply DWT Denoising
Apply K-SVD Denoising
     ↓
Compare Outputs (PSNR, MSE, SSIM)
##  Future Improvements

* Add real-time webcam input
* Integrate deep learning models (DnCNN, Autoencoders)
* Add quantitative metrics dashboard (PSNR, SSIM graphs)
* Improve UI/UX

---

## 👨‍💻 Author

**Granth Choubey**

---

##  Show Your Support

If you found this project useful:

*  Star this repo
*  Fork it
*  Contribute

---

## 📬 Contributions

Pull requests are welcome! Feel free to open issues for suggestions or improvements.
