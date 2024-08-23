Intro
----

A cost focused path based link prediction method, 
the Natural Exponential Similarity Matrix Link Prediction Method (ESMLP), 
uses the exponential function (e^x) expansion form for the similarity matrix expansion.

This method is more suitable for topology networks that focus on reducing costs, 
such as the prediction of oil pipelines and railway lines.

The specific code file can be found in demo.py, 
which compares the AUC indicators of the other five types of link prediction methods with the ESMLP method.


Data Inscriptions
----

1.GZRway: 2022 China Guangzhou Railway Network Data, a total of 22 node samples, manually compiled from the Guangzhou Statistical Yearbook (2023),
Guangzhou Statistical Yearbook (2022) and Guangdong Province's 14th Five-Year Railway Planning Schematic Diagram.
The proximity matrix on the economic distance is calculated by calculating the industrial output value of each city in the training set,
and the median is used to divide whether it is adjacent or not. The test set is a real railway network.

2.HNRway: 2022 Henan railway network data, a total of 17 node samples, manually compiled from the "Henan Statistical Yearbook (2023)",
"Henan Statistical Yearbook (2022)" and "Henan Province "14th Five-Year" Railway Planning Schematic Diagram;
The proximity matrix on the economic distance is calculated by the industrial added value of each city in the training set,
and the median is used to divide whether it is adjacent or not. The test set is a real railway network.

3.CSJRway：2016 railway network in the Yangtze River Delta region of China, a total of 26 node samples, manually compiled from statistical 
yearbooks of certain cities in the Yangtze River Delta region of China.

4.COLoil: interprovincial oil pipeline data in eastern North America, a total of 24 node samples,
manually compiled from the official website of United States Coronal Company (Colonial Pipeline https://www.colpipe.com)
and (Global Covenant of Mayors https://www.globalcovenantofmayors.org);
The proximity matrix on the economic distance is calculated from the oil consumption of each city in the training set.
Use the median to determine whether they are adjacent or not; The test set is a real crude oil pipeline condition.

5.USAir data: United States aviation network USAir data, a total of 332 node samples,
from the paper of Lv Linyuan and Huang Lu et al. (Lv Linyuan, 2010; Huang Lu et al., 2019),
the training set and test set are real route conditions.

Dependencies
----

See requiremets.txt for specific dependencies (Python > 3.6)

Ref
----

Unavailable now.
