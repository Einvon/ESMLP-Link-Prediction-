方法（Intro Inscriptions）
----
(中文)

一种侧重成本的基于路径的链路预测方法，自然指数相似矩阵链路预测方法（ESMLP），对相似矩阵展开式使用了指数函数（e^x）展开形式。

该方法更适用于侧重于降低成本的拓扑网络，如石油管线、铁路线的预测。

具体代码文件请见demo.py，该文件对比了其它五大类链路预测方法与ESMLP方法的AUC指标情况。

（English）

A cost focused path based link prediction method, 
the Natural Exponential Similarity Matrix Link Prediction Method (ESMLP), 
uses the exponential function (e^x) expansion form for the similarity matrix expansion.

This method is more suitable for topology networks that focus on reducing costs, 
such as the prediction of oil pipelines and railway lines.

The specific code file can be found in demo.py, 
which compares the AUC indicators of the other five types of link prediction methods with the ESMLP method.


数据（Data Inscriptions）
----
（中文）

1.GZRway：2022年中国广州铁路网数据，总计22个节点样本，手工整理自《广州统计年鉴（2023）》、《广州统计年鉴（2022）》及《广东省“十四五”铁路规划示意图》。
训练集以各城市工业产值计算得到经济距离上的邻近矩阵，使用中位数划分是否相邻；测试集为真实铁路网络情况。

2.HNRway：2022年河南铁路网数据，总计17个节点样本，手工整理自《河南统计年鉴（2023）》、《河南统计年鉴（2022）》及《河南省“十四五”铁路规划示意图》；
训练集以各城市工业增加值计算得到经济距离上的邻近矩阵，使用中位数划分是否相邻；测试集为真实铁路网络情况。

3.COLoil：北美东部省际输油管道数据，总计24个节点样本，手工整理自美国科罗那尔公司官网数据（Colonial Pipeline https://www.colpipe.com）
及全球市长盟约数据（Global Covenant of Mayors https://www.globalcovenantofmayors.org）
训练集以各城市石油消耗量计算得到经济距离上的邻近矩阵，使用中位数划分是否相邻；测试集为真实原油管道情况。

4.美国航空网络USAir数据，总计332个节点样本，来自吕琳媛及黄璐等人的论文（吕琳媛，2010；黄璐等，2019），训练集与测试集均为真实航线情况。

（English）

1.GZRway: 2022 China Guangzhou Railway Network Data, a total of 22 node samples, manually compiled from the Guangzhou Statistical Yearbook (2023),
Guangzhou Statistical Yearbook (2022) and Guangdong Province's 14th Five-Year Railway Planning Schematic Diagram.
The proximity matrix on the economic distance is calculated by calculating the industrial output value of each city in the training set,
and the median is used to divide whether it is adjacent or not. The test set is a real railway network.

2.HNRway: 2022 Henan railway network data, a total of 17 node samples, manually compiled from the "Henan Statistical Yearbook (2023)",
"Henan Statistical Yearbook (2022)" and "Henan Province "14th Five-Year" Railway Planning Schematic Diagram;
The proximity matrix on the economic distance is calculated by the industrial added value of each city in the training set,
and the median is used to divide whether it is adjacent or not. The test set is a real railway network.

3.COLoil: interprovincial oil pipeline data in eastern North America, a total of 24 node samples,
manually compiled from the official website of United States Coronal Company (Colonial Pipeline https://www.colpipe.com)
and (Global Covenant of Mayors https://www.globalcovenantofmayors.org);
The proximity matrix on the economic distance is calculated from the oil consumption of each city in the training set.
Use the median to determine whether they are adjacent or not; The test set is a real crude oil pipeline condition.

4.USAir data: United States aviation network USAir data, a total of 332 node samples,
from the paper of Lv Linyuan and Huang Lu et al. (Lv Linyuan, 2010; Huang Lu et al., 2019),
the training set and test set are real route conditions.

代码依赖（Dependencies Inscriptions）
----
具体依赖库请见requiremets.txt（Python > 3.6）

See requiremets.txt for specific dependencies (Python > 3.6)

论文参考（Ref）
----
暂无。

Unavailable now.
