import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns



# The official raw data link 
url = "https://raw.githubusercontent.com/ChandniRana/upskill_campus/main/train_aWnotuB.csv"
df=pd.read_csv(url)
print(df.info())
print(df.head(5))
# To checking the trafic  we use  timming of when it was traffic or not by DateTime 
#we have datetime ,junction,id   
# we will convert the datetime into hour,day,month and day of week
# we will use random forest classifier to predict the trafic or not trafic
# firstly we create datetime function then create hour,day,motnh and day of week column which are equals to datetime column and then we will use random forest classifier to predict the trafic or not trafic
df["DateTime"]=pd.to_datetime(df['DateTime'])
df['hour']=df['DateTime'].dt.hour
df['day']=df['DateTime'].dt.day
df['month']=df['DateTime'].dt.month
df['DayOfWeek']=df['DateTime'].dt.dayofweek
df['holiday']=df['DayOfWeek'].apply(lambda x:1 if x >= 5  else 0)
print(df[['DateTime','hour','day','month','DayOfWeek','holiday']].head(5))
plt.figure(figsize=(12,6))
sns.lineplot(data=df,x='hour',y='Vehicles',hue='Junction',palette='viridis')
plt.title("smart city traffic patterns : RUSH HOUR ANALYSIS",fontsize=15)
plt.xlabel("Hours of the day (0-23)",fontsize=12)
plt.ylabel(" Average Vehicle count", fontsize=12)
plt.xticks(range(0,24))
plt.grid(True,alpha=0.3)
plt.show()
