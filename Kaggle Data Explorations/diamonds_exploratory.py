import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sns

data=pd.read_csv("G:\\My Drive\\diamonds.csv")

data=data.drop('Unnamed: 0',axis=1)

print(data.head(5))
print('----------')
print('number of rows:',data.shape[0],'\nnumber of columns:',data.shape[1])
print('----------')
print(data.info())
print('-------------')
print (data.describe())
print('------------------------------')


cut_list=['Premium','Ideal','Very Good','Good','Fair']
data['cut'] = pd.Categorical(data['cut'],categories = cut_list)
data_cut=data.groupby(by=['cut']).mean()
#data_cut.sort_values('cut', inplace=True)
data_cut1=data.groupby(by=['cut'])
print(data_cut)

ax=sns.countplot(data=data,x='cut', order=data['cut'].value_counts().index)
plt.title('cut frequency')
plt.show()

ax1=sns.stripplot(data=data,x='cut',y='price') 
plt.title('cut/price')

plt.show()

print('-----------------')
print('null values:\n',data.isna().sum())
print('-----------------')

print(data.value_counts('color'))

avg_price_by_clarity = data.groupby("clarity")["price"].mean()
#series of the clarity column with mean price per each group
#PAY ATTENTION- .VS.
#data frame with clarity as the row index , colors as colomns and mean price per each intersection 
grouped = data.groupby(["clarity", "color"])["price"].mean().unstack()

plt.figure(figsize=[12,6])
ax3=sns.scatterplot(data=data,x=data.carat,y=data.price,hue=data.cut,marker="x")
plt.xlabel('carat')
plt.ylabel('price')
plt.title('Price vs Carat Analysis')
plt.legend()
sns.move_legend(ax3, "upper left", bbox_to_anchor=(1, 1))

plt.show()


color_arr=pd.unique(data.color)
color_list=color_arr.tolist()
color_list.sort()

plt_color = ["red", "orange", "green", "blue", "purple",'yellow','pink']

plt.figure(figsize=[12, 12])
for i in range(1,len(color_list)+1):
    
    plt.subplot(3,3,i)
    temp_df=data.loc[data['color']==color_list[i-1]]  
    plt.scatter(x=temp_df.carat,y=temp_df.price,color=plt_color[i-1])
    plt.xlabel('carat')
    plt.ylabel('price')
    plt.title('price by carat and color== '+str(color_list[i-1]))
    
plt.suptitle('price by carat and color',fontsize=16)
plt.tight_layout()
plt.show

plt.figure(figsize=[10,10])
ax4=sns.countplot(data=data,x=data.clarity,order=data['clarity'].value_counts().index)
plt.title('clarity frequency')
plt.show()

clarity_list=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
data['clarity'] = pd.Categorical(data['clarity'],categories = clarity_list)

grouped = data.groupby(["clarity", "color"])["price"].mean().unstack()

plt.figure(figsize=[10,10])
colors = ["red", "orange", "green", "blue", "purple", "cyan", "salmon"]
grouped.plot(kind="bar", stacked=True, color=colors, figsize=(12, 6))
plt.xlabel("mean price")
plt.ylabel("clarity")
plt.title("Diamonds: Mean Price by Clarity and Color")
plt.legend(title="Color",bbox_to_anchor=(1, 1),loc='upper left')
plt.tight_layout()
plt.show()
    

plt.figure(figsize=[12,6])
ax=sns.barplot(data=data ,x=data.clarity,y=data.carat)
plt.xlabel('clarity')
plt.ylabel('carat')
plt.title('carat vs clarity Analysis')
plt.show()   

