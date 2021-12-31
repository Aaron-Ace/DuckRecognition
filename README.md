# 實作樸素貝氏分類器用以分辨鴨子像素位置 
---
## 敘述說明
利用分類器將無人機於空中拍攝鴨子圖片(full_duck.jpeg)中的像速做為訓練data，分辨出那些是屬於鴨子的像素，哪些是背景的像素。
並新建一個圖片將背景像素用黑色表示，鴨子部分則用白色表示。
---
## 運算公式
```
P( h | d) = P ( d | h ) * P( h) / P(d)
P ( h | d )：是因子h基於數據d的假設概率,叫做後驗概率
P ( d | h ) : 是假設h為真條件下的數據d的概率
P( h)　: 是假設條件h為真的時候的概率（和數據無關），它叫做h的先驗概率
P(d)　: 數據d的概率，和先驗條件無關．
```
---
##算法實現分解：
* Step 1 : **數據處理**  
加載數據將圖片轉換成訓練數據和測試數據 
* Step 2 : **匯總數據**  
匯總訓練數據的概率以便後續計算概率和做預測 
* Step 3 : **結果預測**  
通過給定的測試圖片和匯總的訓練數據做預測
* Step 4 : **數據轉換圖片**  
將預測結果以黑即白替換並生成圖片 
* Step 5 : **評估準確性**  
使用結果圖來評估預測的準確性     
---   
### Python套件
*   Opencv
*   Numpy
*	Pillow
*	Pandas
```
pip install opencv-python
```
```
pip install numpy
```
```
pip install pillow
```
```
pip install pandas
```
---
## 程式碼簡介

main.py -> 分類程式主架構及輸出(生成結果)

RGBScanner.py ->  將圖片轉換成RGB像素並且輸出(用於訓練資料生成)
---
## Dataset介紹

Dataset 總共大小為 745272 筆訓練資料
其中分類為鴨子像素的為 17442 筆、非鴨子的的分類為 727830 筆
數據比例不成正比是因為背景樣式較多且多元，相較於鴨子的像素單一所以才在數據比例上相當大的差距。

背景範例:
![image_7](https://github.com/Aaron-Ace/DuckRecognition/blob/041f8763aff1ec247da84d07870cea3dd8f7e380/image/background.jpg)
![image_8](https://github.com/Aaron-Ace/DuckRecognition/blob/041f8763aff1ec247da84d07870cea3dd8f7e380/image/background1.jpg)
---
## 成果展示

結果一:
![image_1](https://github.com/Aaron-Ace/DuckRecognition/blob/aecd29b5be79de4b4f376ae012f07a482b948979/result/duck_1.jpeg)

結果二:
![image_2](https://github.com/Aaron-Ace/DuckRecognition/blob/aecd29b5be79de4b4f376ae012f07a482b948979/result/duck_2.jpeg)

結果三:
![image_3](https://github.com/Aaron-Ace/DuckRecognition/blob/aecd29b5be79de4b4f376ae012f07a482b948979/result/duck_3.jpeg)

結果四:
![image_4](https://github.com/Aaron-Ace/DuckRecognition/blob/aecd29b5be79de4b4f376ae012f07a482b948979/result/duck_4.jpeg)

結果五:
![image_5](https://github.com/Aaron-Ace/DuckRecognition/blob/aecd29b5be79de4b4f376ae012f07a482b948979/result/full_duck_1.jpeg)

結果六:
![image_6](https://github.com/Aaron-Ace/DuckRecognition/blob/aecd29b5be79de4b4f376ae012f07a482b948979/result/full_duck_2.jpeg)

##備註:
結果五及六合併為完整full_duck圖片，因素過大，且無法使用GPU運算，所以將圖片切為兩份避免爆RAM及當機
---
#Powered By NDHU CHANG HAO CHAN