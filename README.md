# 實作樸素貝氏分類器用以分辨鴨子像素位置 

[](https://github.com/Aaron-Ace/DuckRecognition)
---
## 敘述說明

利用分類器將無人機於空中拍攝鴨子圖片(full_duck.jpeg)中的像速做為訓練data

分辨出那些是屬於鴨子的像素，哪些是背景的像素

並新建一個圖片將背景像素用黑色表示，鴨子部分則用白色表示。

---
## 運算公式

P( h | d) = P ( d | h ) * P( h) / P(d)
P ( h | d )：是因子h基於數據d的假設概率,叫做後驗概率
P ( d | h ) : 是假設h為真條件下的數據d的概率
P( h)　: 是假設條件h為真的時候的概率（和數據無關），它叫做h的先驗概率
P(d)　: 數據d的概率，和先驗條件無關．

---
## 算法實現分解：
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
```
main.py : 分類程式主架構及輸出(生成結果)
```
```
RGBScanner.py : 將圖片轉換成RGB像素並且輸出(用於訓練資料生成)
```
---
## Dataset介紹

Dataset 總共大小為 **745272** 筆訓練資料，其中分類:
* 鴨子像素的為 **17442** 筆
* 非鴨子的的分類為 **727830** 筆

數據比例不成正比是因為背景樣式較多且多元

相較於鴨子的像素單一所以才在數據比例上相當大的差距。

---
## 背景範例:
<img src="https://i.imgur.com/bthYrYn.jpg" width=100% >
<img src="https://i.imgur.com/0gygFwm.jpg" width=100% >
---
## 成果展示
#### 範例一:
<img src="https://i.imgur.com/qwF6gdk.jpg" width=100% >

#### 結果一:
<img src="https://i.imgur.com/wHJw2xa.jpg" width=100% >

#### 範例二:
<img src="https://i.imgur.com/QKQ0FVw.jpg" width=100% >

#### 結果二:
<img src="https://i.imgur.com/lQ5HPTO.jpg" width=100% >

#### 範例三:
<img src="https://i.imgur.com/hRs1SgW.jpg" width=100% >

#### 結果三:
<img src="https://i.imgur.com/4jCZAUY.jpg" width=100% >

#### 範例四:
<img src="https://i.imgur.com/5ZDbBI8.jpg" width=100% >

#### 結果四:
<img src="https://i.imgur.com/Bcu0Vgp.jpg" width=100% >

#### 範例五:
<img src="https://i.imgur.com/uviCPQ6.jpg" width=100% >

#### 結果五:
<img src="https://i.imgur.com/WBDmhO6.jpg" width=100% >

#### 範例六:
<img src="https://i.imgur.com/rGRdHHT.jpg" width=100% >

#### 結果六:
<img src="https://i.imgur.com/zDmE1nb.jpg" width=100% >

---
## 備註:

結果五及六合併為完整full_duck圖片，因為圖片過大資料太多

且無法使用GPU運算，所以將圖片切為兩份避免爆RAM及當機

---
## 結果討論

根據結果圖來說我認為我的分辨效果非常的良好，僅有部分較大且較白的石頭誤認為鴨子

正確率有至99%以上，我個人認為非常滿意

我認為關鍵的部分為我的資料集數目非常龐大完整

以至結果達到我的預期，相信沒有多少人可以有我這麼大的資料數量

---
## 心得感想:

這次的作業整體而言不難，與機器學習課程的作業一內容幾乎相近

所以修改之前作業的內容即可快速的完成

比較困難的部分在於圖片過大，計算過程又不能使用GPU運算

導致計算的時間蠻長的，所以我將圖片分為二部分，降低失敗的風險

另外要將預測結果轉回圖片也花費我不少心力

對於array的了解要再加強，以便下次運用更加上手。

整體來說我很喜歡這次作業的內容，實務性質高，非常有用。

---
##### Powered By NDHU CHANG HAO CHAN