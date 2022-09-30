# Real Time Violence Detection using MobileNet and Bi-directional LSTM
#### This repository is part of the Graduation Project for NTI Training - AI and IoT DEY Initiative. 
### Kaggle Notebook : https://www.kaggle.com/code/abduulrahmankhalid/real-time-violence-detection-mobilenet-bi-lstm

- ## In this Work, we proposed a real-time violence detector based on deep-learning methods.
  #### The proposed model consists of a MobileNet Pretrained Model as a spatial feature extractor and Bidirectional LSTM as temporal relation learning method with a focus on the three-factor (overall generality - accuracy - fast response time). The suggested model achieved about 97% accuracy with speed of 16 frames/sec.
  ![image](https://user-images.githubusercontent.com/76521677/192987124-6ab45fd6-aef9-4359-a795-c2bbebec674f.png)


- ## Expirements
  ## We Implemented Two Prediction Functions To Test Our Model On
  ### - First Function Perform Frame By Frame Prediction For The Video.

  ![Output-Test-Violence-Video](https://user-images.githubusercontent.com/76521677/192982850-07593c8d-a674-4f2f-a80d-924ae318a9d7.gif)

  ![Output-Test-NonViolence-Video](https://user-images.githubusercontent.com/76521677/192983491-64b20a82-326c-48cb-8932-8e59f8ccdbcc.gif)

  ### - Second Function Perform Prediction For The Whole Video.

    ![image](https://user-images.githubusercontent.com/76521677/192984158-6b942c47-a0a3-409a-9b57-5795b3e548ad.png)
    ![image](https://user-images.githubusercontent.com/76521677/192984193-2a0e11e5-6b2a-4b40-81bc-2227d52853c5.png)

- ## Conclusion
  ### Our proposed MobileNetV2-BiLSTM variant provides the reportedly best results for the used dataset.
  ### Despite the performance of our proposed model, it needs to be further validated with more standard datasets where identification of one to many or many to many violent activities including weapons are tough to detect.


- Refrences: [CNN-BiLSTM Model for Violence Detection in Smart Surveillance](https://link.springer.com/article/10.1007/s42979-020-00207-x#Sec15)
