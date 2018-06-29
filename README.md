# ML_assignment2
Handwritten Digit Recognition

這次的手寫辨識作業在之前的學程中也有用到，所以沒有什麼太大的問題，憑著之前的印象加上查找一點keras的語法很快就完成了。  

model 是 Sequential.
  
所使用的 optimizer 是 RMSprop , 有用到的activation function 是 relu & softmax.
  
原本用三層Dense layer做model，發現測試出的正確率並沒有穩定成長，偶爾上升偶爾下降。  
![image](https://github.com/410421216/ML_assignment2/blob/master/training%20result1.jpg)  
  
加上了兩層Dropout layer以後，狀況有比較改善。
![image](https://github.com/410421216/ML_assignment2/blob/master/training%20result2.jpg)
