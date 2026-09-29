
国庆期间应该完成的一个idea论文初步书写：

时间序列切分patch，计算patch之间的变化，把这些变化利用prototype方法来做原型学习聚类。

假设1 ：变化的类型是多种模式但是种类并没有那么多

然后切分patch的时候包含两种：长期patch和短期patch
利用DBloss的趋势和周期分解的思想，可以对原始的序列做EMA之后再做切分
计算loss的时候包含两个loss：
1. 长期趋势loss
2. 短期patch的预测loss

具体的，使用encoder-preditor-decoder思想来做，先训练en-de，然后加入predictor去预测

不过preditor预测的是从prototype中加权得到的变化原型，这个是一个在latent space的形式
1. 对于短期预测，有一个prototype set，选取之后进行加权
2. 长期也有一个单独的prototype set，也是选取之后加权

然后分解计算loss

最后decoder基于这个变化 去恢复 为原始数值（变化数值），然后把这个数值基于输入来恢复成真实的数值。

此外loss设计上，可以加入latent forcasting里面的，latent space 形状的对齐loss

假设2 ：只计算变化是可行的，参考Nlinear？

其他需要考虑的点：
1. 数据输入时候使用RevIN的方法吗？
2. 短期预测的时候是否需要attention？
3. 长期预测 是否需要引入长期context？ 

每次从原型中是如何选择的？