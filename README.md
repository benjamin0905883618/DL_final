# DL_final - Parkinson’s Disease - LA
本次專案目的是預測**腿部靈敏度**,\
利用課堂提供的訓練資料來訓練出一個可以辨識腿部靈敏度的模型\
訓練資料 : [LA訓練資料](https://140.123.105.254:8282/s/FHTEqYpCQWPjdHk)\
由於訓練的label為UPDRS, 這邊也提供上UPDRS分數的意義\
![](./main/updrs.png =280x257)


## Part 1 : Data preprocessing
在資料上我們可以觀察到幾個問題
* **提供的資料並沒有嚴重程度為4的資料**\
-> 由於沒有類別, 我們預測時沒辦法做這個類別
* **每個影片有兩個類別(左腿跟右腿的UPDRS)**\
-> 根據聯絡人的建議, 採用訓練兩個模型的方式來實作
* **每個類別的資料極不平均**\
-> 只能隨著蒐集資料變多來解決, 這裡無法解決
* **資料缺類別或者類別為X**\
-> 透過資料修剪刪除該筆資料

在解決上述問題後, 我們進一步對資料作處理。

**Openpose**\
由於提供的data是json形式的
![](https://github.com/benjamin0905883618/DL_final/blob/main/keypoint.png)
每三個一組, 前面兩個部分為keypoints座標, 最後一個則是openpose的confidence。
![](https://github.com/benjamin0905883618/DL_final/blob/main/openpose_result.png)\
每個位置則分別代表身體上的不同部位

**Json2imgs**\
由於資料每筆都是一個影片, 而每個json則代表一幀, 因此我們選擇將json資料透過轉換成圖片, 在依訓練需求調整。
透過上述的資料結構我們透過程式將json檔依據影片名稱寫成資料夾, 並將每個影片中的json檔換回圖片。\
由於Openpose也是一個深度學習model, 因此部分幀數可能會有誤測的狀況(由於軀幹距離過近), 這邊也提供誤測的圖, 在訓練中我們並未刪除這類圖片, 只當作躁點希望能些微幫助訓練。\
![](https://github.com/benjamin0905883618/DL_final/blob/75618087c8c48d8400976ddf00a135039b13badb/ques.png)



## Part 2 : ResNet + LSTM
我們透過自定義dataset, 使其自己去讀裝有影片名稱label的xls檔, 並依據所指定的幀數回傳一串影片和其label, 再透過我們自定義的ResNetLSTM去訓練這個model。
```
class ResNetLSTM(nn.Module):
    def __init__(self, resnet_hidden_size, lstm_hidden_size, num_classes):
        super(ResNetLSTM, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # 移除頂層全連接層
        self.resnet_hidden_size = resnet_hidden_size

        self.lstm = nn.LSTM(resnet_hidden_size, lstm_hidden_size, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(-1, C, H, W)

        resnet_output = self.resnet(x)  # 經過 ResNet 特徵提取
        #print(resnet_output.shape)
        resnet_output = resnet_output.view(batch_size, timesteps, self.resnet_hidden_size)
        #print(resnet_output.shape)
        lstm_output, _ = self.lstm(resnet_output)  # 使用 LSTM 進行序列建模
        lstm_output = lstm_output[:, -1, :]  # 取最後一個時間步的輸出

        output = self.fc(lstm_output)  # 進行分類

        return output
```
這個步驟我們希望模型自己去學習該如何擷取特徵, 並且透過LSTM讓他去察覺時序的關係, 進而得出想要的答案。\
但由於label的關係, 我們每筆資料只能選擇同一個影片內的幾幀, batch也沒辦法開太高。
:::warning
由於ResNet + LSTM的效果不如預期, 因此在我們確定左腳的效果並沒辦法將模型train好後我們便沒繼續往下做。
:::
## Part 3 : Video Classification
由於用圖片的訓練方法效果並不好, 因此我們找到了另一種作影片分類的方式, 本次是使用pytorchvideo這個套件來進行資料的處理。

**資料前處理**\
透過程式將圖片預先轉成影片, 並根據類似DatasetFolder的方式將資料預先分類。
![](https://github.com/benjamin0905883618/DL_final/blob/75618087c8c48d8400976ddf00a135039b13badb/data_parse1.png)![](https://github.com/benjamin0905883618/DL_final/blob/75618087c8c48d8400976ddf00a135039b13badb/data_parse2.png)
再透過pytorchvideo的套件讀成Iter-Dataset。
由於轉成這個型態後, 無法使用Random_split的套件, 因此我們只能預先將訓練集和驗證集分割好, 由於類別3的影片只有兩支, 所以我們在驗證集在這個類別會用複製的方式, 避免影響訓練的效果, 大約擷取10支影片作驗證, 大約是7:3的比例去做訓練和驗證。

**訓練**\
使用torchvision的video_model.r3d_18及其預訓練權重進行finetune。
![](https://github.com/benjamin0905883618/DL_final/blob/75618087c8c48d8400976ddf00a135039b13badb/hint_model_L/L_loss_surface.png)![](https://github.com/benjamin0905883618/DL_final/blob/75618087c8c48d8400976ddf00a135039b13badb/hint_model_R/R_loss_surface.png)\
由於資料太少, 用同樣的資料和模型, 但不同的分割train、valid的資料, 導致截然不同的結果。\
在訓練中, 我們採用了Data Hint的方法些微增加資料, 在擴增資料集中, 我們除了原本的Normalize, 只加上了RandomRotation來幫助訓練, 並透過pytorchvideo本身的套件進行"**uniform**"片段擷取, 每次約擷取5秒。

**測試**\
在測試中, 除了資料過少的類別3外, 皆是來自valid_set, 但我們將擷取的長度改為8秒, 並且改用隨機擷取片段的方式來進行測試。

