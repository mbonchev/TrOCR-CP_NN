<h1 align="center"><font size="10"><b>Course Project on Neural Networks</b></font></h1>

**• Topic: Transformer-based Optical Character Recognition<br><br>
• Student: Martin Bonchev<br>
• Faculty number: 121222203</br></br>
• Faculty: FCST (ФКСТ)<br>
• Specilty: CSI (КСИ)<br>
• Group: 37**
<hr>

### • Dataset - [standard OCR dataset](https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset/data)
&emsp;- "*data2/testing_data*" is used as validation data ("*data/validation_data*"). For easier file transfer, a zip file of the dataset (with the just mentioned validation data) is attached. 
### • Model - [trocr-large-printed](https://huggingface.co/microsoft/trocr-large-printed)
### • Environment - [Google Colab](https://colab.google/)
<hr>

&emsp;&emsp;Initially, I tried to fine-tune the model locally on my PC, but I found out that my GPU (RX 580) was not capable of doing that due to incompatibilities with PyTorch, which means that only the CPU was used. After many attempts and several (~10) PC crashed, I finally realized this task was not feasible on my machine due to HW limitations.<br><br>
&emsp;&emsp;After researching for other ways to accompolish the job and overcome the obstacle, I decided to switch to Google Colab to perform the fine-tuning. I also used Weights & Biases (WandB), which allowed me to log metrics, visualize training progress, and keep a record of my runs efficiently.
