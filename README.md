# Customer_Churn_Prediction

Contents of this document – 
1.	Objective 
2.	About the dataset 
3.	Python libraries used in the project 
a.	NumPy 
b.	Pandas 
c.	Scikit-Learn 
d.	Matplotlib 
e.	Seaborn 
f.	Scipy 
4.	Load dataset 
5.	Exploratory data analysis (Checking missing values, duplicated values, unbalanced classes) 
6.	Data preprocessing - Handling missing values, Categorical encoding - (Label encoding, On-hot encoding) 
7.	Data visualization - Univariate analysis, Countplot, Barplot, Checking correlation using heatmap 
8.	Train test split 
9.	Machine learning models used - Decision Tree Classifier. Random Forest Classifier, XGBoost Classifier.
    Logistic Regression, PCA
10.	Model evaluation methods used - Score – accuracy score, Classification report (Precision, Recall, F1 score,
    Support, Accuracy, Macro average, Weighted average), Confusion matrix
11.	SMOTEENN method 
12.	Cross validation 
13.	RandomizedSearchCV 
14.	AUC-ROC Curve
15.	Model built with 96.87% accuracy score and 99.45% AUC-ROC score

About the problem statement / objective – 

Objective is to build a predictive model that can predict customer churn for a given company. Use machine learning techniques to build the model and document the 
process, including feature selection, model evaluation, and performance metrics. 

About the dataset – 

Dataset used - https://www.kaggle.com/competitions/bank-marketing-uci/overview

The data is related with direct marketing campaigns of a Portuguese banking institution. 

Bank client data -
1.	Age (numeric)
2.	Job – job type (categorical:
'admin.','bluecollar','entrepreneur','housemaid','management','retired','selfemployed','services','student','technician','unemployed','unknown')
3.	Marital – marital status (categorical : 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4.	Education - (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5.	Default - has credit in default (categorical: 'no','yes','unknown')
6.	Housing - has housing loan (categorical: 'no','yes','unknown')
7.	Loan – has personal loan (categorical: 'no','yes','unknown')
8.	Contact - contact communication type (categorical: 'cellular','telephone')
9.	Month -  last contact month of year (categorical: 'jan', 'feb', 'mar', …, 'nov', 'dec')
10.	day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11.	duration: last contact duration, in seconds (numeric).
12.	campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13.	pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14.	previous: number of contacts performed before this campaign and for this client (numeric)
15.	poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
16.	y - has the client subscribed a term deposit? (binary: 'yes','no')

Python libraries used in the project – 

1.	NumPy – 
It is a python library used for working with arrays. It has functions for working in the domain of linear algebra, fourier transform, and matrices. It is the 
fundamental package for scientific computing with python. NumPy stands for numerical python. 
NumPy is preferred because it is faster than traditional python lists. It has supporting functions that make working with ndarray very easy. Arrays are frequently used 
where speed and resources are very important. NumPy arrays are faster because it is stored at one continuous place in memory unlike lists, so processes can access and 
manipulate them very efficiently. This is locality of reference in computer science.
2.	Pandas – 
Pandas is made for working with relational or labelled data both easily and intuitively. It provides various data structures and operations for manipulating numerical
data and time series. 
Pandas is built on top of NumPy library. That means that a lot of structures of NumPy are used or replicated in Pandas. The data produced by pandas are often used as 
input for plotting functions of Matplotlib, statistical analysis in SciPy, and machine learning algorithms in Scikit-learn.
It has a lot of advantages like –
a.	Fast and efficient for manipulating and analyzing data
b.	Data from different file objects can be loaded 
c.	Easy handling of missing data in data preprocessing 
d.	Size mutability 
e.	Easy dataset merging and joining 
f.	Flexible reshaping and pivoting of datasets 
g.	Gives time-series functionality
3.	Scikit-Learn – 
It provides efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction. It has numerous 
machine learning, pre-processing, cross validation, and visualization algorithms.
4.	Matplotlib – 
It is used for 2D array plots. It includes wide range of plots, such as scatter, line, bar, histogram and others that can assist in delving deeper into trends.
5.	Seaborn – 
Seaborn aims to make visualization the central part of exploring and understanding data. It provides dataset-oriented APIs, so that we can switch between different 
visual representations for same variables for better understanding of dataset. It is 
used for creating statistical representations based on datasets. It is built on top of matplotlib. It is built on top of pandas’ data structures. The library conducts 
the necessary modelling and aggregation internally to create insightful visuals. 
6.	SciPy – 
Stands for Scientific Python. It is a scientific computation library that uses NumPy. It provides more utility functions for optimization, stats and signal processing. These optimized and added functions are commonly used in NumPy and data science. 

Exploratory data analysis – 

Exploratory data analysis is the process of performing initial investigation on the data to discover patterns or spot anomalies. It is done to test the hypothesis and 
to check assumptions with the help of summary statistics and graphical representations.

‘describe()’ method returns description of data in DataFrame. It tells us the following information for each column - 
1.	Count - number of non-empty values
2.	Mean - the average (mean) value  
3.	Std - standard deviation
4.	Min - minimum value
5.	25% - the 25 percentile 
6.	50% - the 50 percentile 
7.	75% - the 75 percentile
8.	Max - maximum value

The info() method prints the information about dataframe. 

It contains the number of columns, column labels, column data types, memory usage, range index, and number of cells in each column. 

Parameters - 
1.	verbose - It is used to print the full summary of the dataset.
2.	buf - It is a writable buffer, default to sys.stdout.
3.	max_cols - It specifies whether a half summary or full summary is to be printed.
4.	memory_usage - It specifies whether total memory usage of the DatFrame elements (including index) should be displayed.
5.	null_counts - It is used to show the non-null counts.

From the output, we see there are 7 features with ‘int’ value attributes and 10 with ‘object’ data type. 

value_counts() function - 

It is used to get a series containing counts of unique values. 

Parameters - 
1.	Normalize - If True then the object returned will contain the relative frequencies of the unique values.	
2.	Sort - Sort by frequencies.	
3.	Ascending - Sort in ascending order.
4.	Bins - Rather than count values, group them into half-open bins, only works with numeric data.

Unbalanced classes - 

In classification cases, when the data available on one or more classes are extremely low, then it is an unbalanced class. 

This can be a problem because - 
1.	We don’t get optimized results for the class which is unbalanced in real time as the algorithm model does not get sufficient insight at the underlying class. 
2.	It creates a problem in making validation to test data because it is difficult to have representation across classes in case number of observations for few classes 
is extremely less. 

Following are some of the ways of handling it - 
1.	Undersampling - Here, we randomly delete the class which has sufficient observations so that the comparative ration of two classes is significant in our data. This 
approach is simple but it can introduce a bias in the data because there is a high possibility that the data we are deleting may contain important information about the 
predictive class. 
2.	Oversampling - For the unbalanced class randomly increase the number of observations which are just copies of existing samples. This ideally gives a sufficient 
number of samples to work with. However, oversampling may lead to overfitting to the training data. 
3.	Synthetic sampling - synthetically manufacture observations of unbalanced classes which are similar to the existing using nearest neighbour classification. The 
problem comes when the number of observations are of extremely rare class.	

Now that we have studied out dataset thoroughly, we can start with our data preprocessing, data cleaning methods. 

Missing values - 

Missing values are common when working with real-world datasets. Missing data could result from a human factor, a problem in electrical sensors, missing files, 
improper management or other factors. Missing values can result in loss of significant information. Missing value can bias the results of model and reduce the accuracy 
of the model. There are various methods of handling missing data but unfortunately they still introduce some bias such as favoring one class over the other but these 
methods are useful. 
In Pandas, missing values are represented by NaN. It stands for Not a Number. 

Reasons for missing values - 
1.	Past data may be corrupted due to improper maintenance
2.	Observations are not recorded for certain fields due to faulty measuring equipments. There might by a failure in recording the values due to human error. 
3.	The user has not provided the values intentionally. 

Why we need to handle missing values - 
1.	Many machine learning algorithms fail if the dataset contains missing values. 
2.	Missing values may result in a biased machine learning model which will lead to incorrect results if the missing values are not handled properly. 
3.	Missing data can lead to lack of precision. 

Types of missing data - 

Understanding the different types of missing data will provide insights into how to approach the missing values in the dataset. 
1.	Missing Completely at Random (MCAR) 
There is no relationship between the missing data and any other values observed or unobserved within the given dataset. Missing values are completely independent of 
other data. There is no pattern. The probability of data being missing is the same for all the observations. 
The data may be missing due to human error, some system or equipment failure, loss of sample, or some unsatisfactory technicalities while recording the values.
It should not be assumed as it’s a rare case. The advantage of data with such missing values is that the statistical analysis remains unbiased.   
2.	Missing at Random (MAR)
The reason for missing values can be explained by variables on which complete information is provided. There is relationship between the missing data and other 
values/data. In this case, most of the time, data is not missing for all the observations. It is missing only within sub-samples of the data and there is pattern in 
missing values. 
In this, the statistical analysis might result in bias. 
3.	Not MIssing at Random (NMAR)
Missing values depend on unobserved data. If there is some pattern in missing data and other observed data can not explain it. If the missing data does not fall under 
the MCAR or MAR then it can be categorized as MNAR. 
It can happen due to the reluctance of people in providing the required information. 
In this case too, statistical analysis might result in bias. 

How to handle missing values - 

isnull().sum() - shows the total number of missing values in each columns 

We need to analyze each column very carefully to understand the reason behind missing values. There are two ways of handling values - 
1.	Deleting missing values - this is a simple method. If the missing value belongs to MAR and MCAR then it can be deleted. But if the missing value belongs to MNAR 
then it should not be deleted. The disadvantage of this method is that we might end up deleting useful data. 
You can drop an entire column or an entire row. 
2.	Imputing missing values - there are various methods of imputing missing values
a.	Replacing with arbitrary value 
b.	Replacing with mean - most common method. But in case of outliers, mean will not be appropriate
c.	Replacing with mode - mode is most frequently occuring value. It is used in case of categorical features. 
d.	Replacing with median - median is middlemost value. It is better to use median in case of outliers. 
e.	Replacing with previous value - it is also called a forward fill. Mostly used in time series data. 
f.	Replacing with next value - also called backward fill. 
g.	Interpolation

Next step is categorical encoding.

Datasets contain multiple labels (in the form of words of numbers) in one or more than one columns. Training data is often labelled in words to make the data more 
human-readable and understandable. Categorical data encoding is coding data to convert it to a machine-readable form. It is an important data preprocessing method. 
There are two types of categorical data 
1.	Ordinal data - categories have an inherent order
2.	Nominal data - no inherent order of categories

Label encoding - 

Label encoding refers to converting labels into numeric form to convert it to machine-readable form. It is an important data preprocessing method for structured 
dataset in supervised learning. 
Limitation - label encoding assigns a unique number to each class of data. However, this can lead to generation of priority issues in the training of data. A label 
with high value may be considered to have high priority than a label having a lower value.

One-hot encoding – 

Label encoding is straight but it has disadvantage – the numeric values can be misinterpreted by algorithms as having some sort of hierarchy/order in them. This 
disadvantage can be overcome by using one-hot encoding. In this strategy, each category value is converted into a new column and assigned a 1 or 0 (for true of false) 
value to the column. 
This method eliminates the hierarchy/order issues but has a disadvantage too. It adds more columns to the dataset. It causes the number of columns to expand greatly 
if you have many unique values in a category column. 

Data visualisation - 

Datasets often come in csv files, spreadsheets, table form etc. Data visualisation provides a good and organized pictorial representation of data which makes it easier 
to observe, understand and analyze.

Univariate analysis – 

Univariate Analysis is a type of data visualization where we visualize only a single variable at a time. Univariate Analysis helps us to analyze the distribution of 
the variable present in the data so that we can perform further analysis. It doesn’t deal with causes or relationships. It’s function is to describe; take data, 
summarize the data and find patterns in the data. 

Countplot - 

seaborn.countplot() method is used to show the counts of observations in each categorical bin using bars.

Parameters - 
1.	x, y - This parameter take names of variables in data or vector data, optional, Inputs for plotting long-form data.
2.	Hue - (optional) This parameter take column name for colour encoding.
3.	Data - (optional) This parameter take DataFrame, array, or list of arrays, Dataset for plotting. If x and y are absent, this is interpreted as wide-form. Otherwise 
it is expected to be long-form.
4.	order, hue_order - (optional) This parameter take lists of strings. Order to plot the categorical levels in, otherwise the levels are inferred from the data objects.
5.	Orient - (optional)This parameter take “v” | “h”, Orientation of the plot (vertical or horizontal). This is usually inferred from the dtype of the input variables but 
can be used to specify when the “categorical” variable is a numeric or when plotting wide-form data.
6.	Color - (optional) This parameter take matplotlib color, Color for all of the elements, or seed for a gradient palette.
7.	Palette - (optional) This parameter take palette name, list, or dict, Colors to use for the different levels of the hue variable. Should be something that can be 
interpreted by color_palette(), or a dictionary mapping hue levels to matplotlib colors.
8.	Saturation - (optional) This parameter take float value, Proportion of the original saturation to draw colors at. Large patches often look better with slightly 
desaturated colors, but set this to 1 if you want the plot colors to perfectly match the input color spec.
9.	Dodge - (optional) This parameter take bool value, When hue nesting is used, whether elements should be shifted along the categorical axis.
10.	Ax - (optional) This parameter take matplotlib Axes, Axes object to draw the plot onto, otherwise uses the current Axes.
11.	Kwargs - This parameter take key, value mappings, Other keyword arguments are passed through to matplotlib.axes.Axes.bar().
12.	Returns - Returns the Axes object with the plot drawn onto it.

Barplot - 

A barplot is used for categorical data according to some methods and by default it’s the mean. It can also be understood as a visualization of the group by action. To 
use this plot we choose a categorical column for the x-axis and a numerical column for the y-axis, and we see that it creates a plot taking a mean per categorical 
column.
It has similar parameters.

corr() - 

corr() method finds the pairwise correlation of each column in a dataframe. 
There are two types of correlation - 
1.	Positive correlation - when the columns are directly proportional to each other 
2.	Negative correlation - when the columns are inversely proportional to each other.

Heatmap - 

Heatmap is graphical representation of data using colors to visualize value of the matrix. It is a plot of rectangular data as color-encoded matrix. 

Parameters:
1.	data: 2D dataset that can be coerced into an ndarray.
2.	vmin, vmax: Values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments.
3.	cmap: The mapping from data values to color space.
4.	center: The value at which to center the colormap when plotting divergent data.
5.	annot: If True, write the data value in each cell.
6.	fmt: String formatting code to use when adding annotations.
7.	linewidths: Width of the lines that will divide each cell.
8.	linecolor: Color of the lines that will divide each cell.
9.	cbar: Whether to draw a colorbar.

Train test split – 

The entire dataset is split into training dataset and testing dataset. Usually, 80-20 or 70-30 split is done. The train-test split is used to prevent the model from 
overfitting and to estimate the performance of prediction-based algorithms. We need to split the dataset to evaluate how well our machine learning model performs. The 
train set is used to fit the model, and statistics of training set are known. Test set is for predictions. 
This is done by using scikit-learn library and train_test_split() function. 

Parameters - 
1.	*arrays: inputs such as lists, arrays, data frames, or matrices
2.	test_size: this is a float value whose value ranges between 0.0 and 1.0. it represents the proportion of our test size. its default value is none.
3.	train_size: this is a float value whose value ranges between 0.0 and 1.0. it represents the proportion of our train size. its default value is none.
4.	random_state: this parameter is used to control the shuffling applied to the data before applying the split. it acts as a seed.
5.	shuffle: This parameter is used to shuffle the data before splitting. Its default value is true.
6.	stratify: This parameter is used to split the data in a stratified fashion.

Decision Tree Classifier – 

Decision tree is a supervised learning technique. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the 
decision rules and each leaf node represents the outcome. 
In a decision tree, there are essentially two nodes – decision node and leaf node. Decision node is used to make decision and have multiple branches. Leaf nodes are 
output of those decisions and do not have any further branches. 
The decisions or the test are performed on the basis of features of the given dataset. It is a graphical representation for getting all the possible solutions to a 
problem/decision based on given conditions. It is called a decision tree because, similar to a tree, it starts with the root node, which expands on further branches 
and constructs a tree-like structure.
I am using decision tree classifier because it usually mimic human thinking ability while making a decision, so it is easy to understand. The logic behind the 
decisions can be easily understood. It shows all the possible outcomes for a problem. There is less requirement of data cleaning compared to other algorithms. 
However, there are a few disadvantages too – it contains a lot of layers, which can become complex to understand sometimes. It may have an overfitting problem. 

Model evaluation - 

Model evaluation is done to test the performance of machine learning model. It is done to determine whether the model is a good fit for the input dataset or not. 

Model evaluation methods used – 

Score method – 

It takes the feature matrix (x_test) and the expected target values (y_test). Prediction for x_test are compared with y_test and either accuracy or R2 score is given. 
Since, we are using a classifier, it gives the accuracy score. 

Classification report – 

Report that explains everything about the classification. It is a summary of the quality of classification made by the ML model. 
It comprises of 5 columns and (n+3) rows. The first column is the class label’s name, followed by Precision, Recall, F1-score, and Support. N rows are for N class labels
and other three rows are for accuracy, macro average, and weighted average.
1.	Precision: It is calculated with respect to the predicted values. For class-A, out of total predictions how many were really belong to class-A in actual dataset, 
is defined as the precision. It is the ratio of [i][i] cell of confusion matrix and sum of the [i] column.
2.	Recall: It is calculated with respect to the actual values in dataset. For class-A, out of total entries in dataset, how many were actually classified in class-A 
by the ML model, is defined as the recall. It is the ratio of [i][i] cell of confusion matrix and sum of the [i] row.
3.	F1-score: It is the harmonic mean of precision and recall.
4.	Support: It is the total entries of each class in the actual dataset. It is simply the sum of rows for every class-i.

Confusion matrix – 

A Confusion matrix is an N x N matrix used for evaluating the performance of a classification model, where N is the number of target classes. The matrix compares the 
actual target values with those predicted by the machine learning model. This tells how well our classification model is performing and what kinds of errors it is 
making.
1.	The target variable has two values – positive and negative 
2.	The columns represent the actual values of the target variable 
3.	The rows represent the predicted values of the target variable. 

SMOTEENN – 

SMOTE – ENN 

Combining SMOTE and ENN (edited nearest neighbour) method

Combining undersampling and oversampling – 
This method combines the SMOTE ability to generate synthetic examples for minority class and ENN ability to delete some observations from both classes that are 
identified as having different class between the observation’s class and its K-nearest neighbor majority class.
1.	(Start of SMOTE) Choose random data from the minority class.
2.	Calculate the distance between the random data and its k nearest neighbors.
3.	Multiply the difference with a random number between 0 and 1, then add the result to the minority class as a synthetic sample.
4.	Repeat step number 2–3 until the desired proportion of minority class is met. (End of SMOTE)
5.	(Start of ENN) Determine K, as the number of nearest neighbors. If not determined, then K=3.
6.	Find the K-nearest neighbor of the observation among the other observations in the dataset, then return the majority class from the K-nearest neighbor.
7.	If the class of the observation and the majority class from the observation’s K-nearest neighbor is different, then the observation and its K-nearest neighbor are 
deleted from the dataset.
8.	Repeat step 2 and 3 until the desired proportion of each class is fulfilled. (End of ENN)

Using SMOTEENN, the accuracy score increases. 

Random forest classifier – 

It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the 
model.
Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy 
of that dataset. Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it 
predicts the final output. The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.

Why I am using Random Forest?
1.	It takes less training time as compared to other algorithms.
2.	It predicts output with high accuracy, even for the large dataset it runs efficiently.
3.	It can also maintain accuracy when a large proportion of data is missing.
4.	It is capable of handling large datasets with high dimensionality.
5.	It enhances the accuracy of the model and prevents the overfitting issue.

Hyperparameters - 
1.	n_estimators– number of trees the algorithm builds before averaging the predictions.
2.	max_features– maximum number of features random forest considers splitting a node.
3.	mini_sample_leaf– determines the minimum number of leaves required to split an internal node.
4.	n_jobs– it tells the engine how many processors it is allowed to use. If the value is 1, it can use only one processor but if the value is -1 there is no limit.
5.	random_state– controls randomness of the sample. The model will always produce the same results if it has a definite value of random state and if it has been given 
the same hyperparameters and the same training data.
6.	oob_score – OOB means out of the bag. It is a random forest cross-validation method. In this one-third of the sample is not used to train the data instead used to 
evaluate its performance. These samples are called out of bag samples.

XGBoost Classifier – 

It is an optimized distributed gradient boosting library designed for efficient and scalable training of machine learning models. It is an ensemble learning method 
that combines the predictions of multiple weak models to produce a stronger prediction.

Hyperparameters – 
1.	learning_rate: step size shrinkage used to prevent overfitting. Range is [0,1]
2.	max_depth: determines how deeply each tree is allowed to grow during any boosting round.
3.	subsample: percentage of samples used per tree. Low value can lead to underfitting.
4.	colsample_bytree: percentage of features used per tree. High value can lead to overfitting.
5.	n_estimators: number of trees you want to build.
6.	objective: determines the loss function to be used like reg:linear for regression problems, reg:logistic for classification problems with only decision, 
binary:logistic for classification problems with probability.

Why I am using XGBoost Classifier –
1.	Great speed and performance
2.	Core algorithm is parallelizable 
3.	Shows better result than other algorithms 
4.	Has wide variety of tuning parameters as seen above. 

Logistic regression – 
Regression models describe the relationship between variables by fitting a line to the observed data. Linear regression models use a straight line and logistic and 
non-linear regression models use a curved line. Regression allows to estimate how a dependent variable changes as the independent variables change. 
Logistic regression (or sigmoid function or logit function) is a type of regression analysis and is commonly used algorithm for solving binary classification problems. 
It predicts a binary outcome based on a series of independent variables. The output is a predicted probability, binary value rather than numerical value. If the 
predicted value is a considerable negative value, it’s considered close to zero. If the predicted value if a significant positive value, it’s considered close to one. 
The dependent variable generally follows bernoulli distribution. Unlike linear regression model, that uses ordinary least square for parameter estimation, logistic 
regression uses maximum likelihood estimation, gradient descent and stochastic gradient descent. There can be infinite sets of regression coefficients. The maximum 
likelihood estimate is that set of regression coefficients for which the probability of getting data we have observed is maximum. To determine the values of parameters, 
log of likelihood function is taken, since it does not change the properties of the function. The log-likelihood is differentiated and using iterative techniques like 
newton method, values of parameters that maximise the log-likelihood are determined. A confusion matrix may be used to evaluate the accuracy of the logistic regression 
algorithm

PCA – 
It is an unsupervised learning algorithm that is used for the dimensionality reduction in machine learning. It is a statistical process that converts the observations 
of correlated features into a set of linearly uncorrelated features. These new transformed features are called the Principal Components. It is a technique to draw 
strong patterns from the given dataset by reducing the variances.
Steps of PCA – 
1.	Representing data into structure 
2.	Standardizing the data 
3.	Calculating the covariance of z 
4.	Calculating the eigen values and eigen vectors 
5.	Sorting the eigen vectors 
6.	Calculating new features or principal components 
7.	Remove less important features from the new dataset. 
I am using PCA because it simplifies the complexity in high-dimensional data while retaining the trends and patterns. 

XGBoost Classifier shows the best result. Therefore, we will finalise that model and finetune it. 

Cross validation – 

Cross-validation is a technique for validating the model efficiency by training it on the subset of input data and testing on previously unseen subset of the input
data.
I am using k-fold cross validation - K-fold cross-validation approach divides the input dataset into K groups of samples of equal sizes. These samples are called 
folds. For each learning set, the prediction function uses k-1 folds, and the rest of the folds are used for the test set.

RandomizedSearchCV – 

A fixed number of parameter settings is sampled from the specified distributions. The major benefit is that it has decreased processing time. 

AUC – ROC – 

It is an evaluation matrix used to visualize performance of classification model. 
ROC curve – Receiver Operating Characteristic curve gives probability graph to show the performance of a classification model at different threshold levels. The curve 
is plotted between two parameters – true positive rate (y axis), false positive rate (x-axis). 
AUC – area under the curve. It calculates the two-dimensional area under the entire ROC curve ranging from (0,0) to (1,1). 
If auc-roc value is closer to 1, it is considered a good model as it shows a good measure of separability. 
