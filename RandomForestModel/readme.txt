Training:

	The python file yash_train.py is used for training a RandomForestModel
	on the cleaned dataset stored in the folder data/

	yash_train.py generates two files yash_classifier and yash_ pca
	These are pickled classifier and pca models

	stdout of yash_train.py:

		yash@yash-Inspiron-7560:~/Documents/Me/ASU/DM/project2$ python yash_train.py 
		/usr/local/lib/python3.5/dist-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
		  from pandas.core import datetools
		/home/yash/.local/lib/python3.5/site-packages/sklearn/ensemble/weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
		  from numpy.core.umath_tests import inner1d
		    0    1    2    3    4    5   ...       24     25     26     27     28     29
		0  221  232  240  244  245  246  ...    245.0  242.0  239.0  235.0  231.0  226.0
		1   55   62   70   79   89  100  ...    144.0  139.0  138.0  138.0  139.0  139.0
		2  214  213  211  210  205  200  ...     79.0   79.0   81.0   82.0   84.0   86.0
		3  226  224  228  222  214  206  ...    143.0  148.0  149.0  148.0  153.0  155.0
		4  188  181  176  174  172  170  ...    100.0   99.0  100.0  101.0  100.0   95.0

		[5 rows x 30 columns]
		(463, 30)
		Confusion Matrix:
		[[35 10]
		 [16 32]]
			     precision    recall  f1-score   support

			  0       0.69      0.78      0.73        45
			  1       0.76      0.67      0.71        48

		avg / total       0.73      0.72      0.72        93

		Accuracy:0.7204301075268817


Testing:

	The python file yash_test.py is used for loading the stored models
	yash_classifier and yash_pca and providing inference on the given CSV test file

	Note: yash_test.py takes command line argument for input test CSV file

	Example Command: python yash_test.py data/Nomeal3.csv 

Also included DM_Asg2.ipynb a jupyter notebook for studying the outputs

requirements.txt file is for installing required python libraries
