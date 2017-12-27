## 人脸识别
### 实验环境
- python2.7
- openCV 2.4.13

人脸识别程序主要分为三个部分：
1. 创建一个dataset：datasetCreator.py
2. 训练识别器：trainner.py
3. 检测器：detector.py

### dataset
通过摄像头每隔0.1s检测一次人脸，并将检测到的人脸储存至dataSet路径下

### trainner
使用openCV自带的LBP人脸特征值提取方法对dataSet路径下的所有人脸进行识别训练

生成训练文件trainningData.yml存放在recognizer路径下

### detector
使用训练后的识别器所产生的trainningData.yml创建一个识别器

利用该识别器对摄像头采集的图像进行逐帧识别

若识别到的人脸储存在dataSet中，显示对应的人名

## 使用方法
1. 运行datasetCreator.py 输入人脸编号，默认采集50张样本
2. 运行trainner.py 训练识别器
3. 运行detector.py 打开摄像头识别人脸