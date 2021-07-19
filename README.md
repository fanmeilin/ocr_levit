### 判断字符缺陷

- 识别轴承上的若干字符串，与给定的pattern匹配。返回：是否为NG，详细信息

> 目前pattern处理的是给定的字符串形式

#### 使用方法
- 定义classes名称和权重路径建立Word_Classification实例，通过图像信息和匹配字符串调用该实例的get_str_matchInfo函数，可以得到ng判断，以及具体的str_bbox_list，str_list信息

```python
from character_ng_detect import Word_Classification

distribution_classes = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
weights_path = "./assets/resnet34.pt"

start = time.time()

word_classifier = Word_Classification(weights_path,distribution_classes)
is_NG,result = word_classifier.get_str_matchInfo(img,bbox_list,r_inner,r_outer,center,pattern_list)

end = time.time()

print("last: ",end-start)
print("is_NG:",is_NG)
print("info:",result)
```