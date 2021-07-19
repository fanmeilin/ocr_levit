### 判断字符缺陷

- 识别轴承上的若干字符串，与给定的pattern匹配。返回：是否为NG，详细信息

> 目前pattern处理的是给定的字符串形式

#### 使用方法
- 定义classes名称和权重路径建立Word_Classification实例，通过图像信息和匹配字符串调用该实例的get_str_matchInfo函数，可以得到ng判断，以及具体的str_bbox_list，str_list信息

```python
from orcA_13 import Word_Classification

start = time.time()

word_classifier = Word_Classification(gpu_id=0)
is_NG,result = word_classifier.get_str_matchInfo(img,bbox_list,r_inner,r_outer,center,pattern_list)

end = time.time()

print("last: ",end-start)
print("is_NG:",is_NG)
print("info:",result)
```