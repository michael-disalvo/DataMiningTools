# preprocessor.py

### How to Install 
Clone repo 
```
git clone https://github.com/michael-disalvo/DataMiningTools.git
```
Add preprocessor.py to your current directory. 

### How to Use
Initialize preprocessor
```python
from preprocessor import preprocessor
preprocessor = preprocessor()
```
Get a summary of dataframe
```python
preprocessor.summary(data) 
```
Train Preprocessor
```python
'''
    # num_std: used in outlier removal
    # variance: used in low variance feature removal
    # correlation: used in high correlation feature removal
'''
dict = {"num_std": 8, 'variance':1e-4, 'correlation':.9}
preprocessor.fit(dict)
```
Transform Data
```python
processed_data = preprocessor.transform(data)
```
### Description
preprocessor is a class that can be used to summarize, clean rows, and select features in a pandas dataframe. 
The structure of the class allows the user to train the processor with certain cleaning parameters, and offers 
feedback for tuning. Once parameters are set, the processor can be used to transform any data set with the 
same settings. 
