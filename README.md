
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
streamlit run app.py
```

---

##  Demo

>  <img width="1919" height="1079" alt="Screenshot 2026-04-11 004616" src="https://github.com/user-attachments/assets/fc56d41f-db91-4cfc-a769-aa0851f5b916" />
><img width="1905" height="944" alt="Screenshot 2026-04-10 225131" src="https://github.com/user-attachments/assets/1aaaee89-8b8e-499d-be19-21563bfa1cbb" />
><img width="1449" height="759" alt="Screenshot 2026-04-10 225212" src="https://github.com/user-attachments/assets/18a7dec9-18f0-433a-b63f-74dff0afeb06" />
><img width="1426" height="903" alt="Screenshot 2026-04-10 225244" src="https://github.com/user-attachments/assets/c077a54f-b76d-41fc-9ee3-a4da03a67b32" />
><img width="1671" height="624" alt="Screenshot 2026-04-10 225739" src="https://github.com/user-attachments/assets/11591ddf-dcec-4048-bdb8-aafa37f933d8" />

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
