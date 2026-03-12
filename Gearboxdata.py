import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
healthy=pd.read_csv("/home/sonu-nitu/Downloads/gearboxdata (1)/healthy.csv/h30hz0.txt",sep='\s+',header=None)
broken=pd.read_csv("/home/sonu-nitu/Downloads/gearboxdata (1)/broken.csv/b30hz0.txt",sep='\s+',header=None)

#name the 4 sensors for both dataframes 
columns=['sensor1','sensor2','sensor3','sensor4']
healthy.columns=columns
broken.columns=columns  

#creating graph to compare them 
plt.figure(figsize=(15,6))

#ploting for healty
plt.subplot(2,1,1)#1 row, 2 column, 1st plot
plt.plot(healthy['sensor1'].iloc[:1000],label='sensor1')
plt.title("Healthy Data")
plt.grid(True)




#plot for broken

plt.subplot(2,1,2)#1 row, 2 column, 2nd plot
plt.plot(broken['sensor1'].iloc[:1000],label='sensor1',color='red')
plt.title("Broken Data")
plt.grid(True)
plt.tight_layout()#adjust the spacing between subplots

plt.show()

#training a model to classify healthy and broken data

# we use function named get feature thats why we dont need to acces both dataset methods one by one 
def get_features(file_path,label):#both files name accessable in one variable
    df=pd.read_csv(file_path,sep='\s+')

    
    #now we check the the  is working perfetly or you can say healthy or broken ,done by calculating MEANS(middle point of vibration),or STD(up down of the vibration)
    features= df.mean().tolist()+df.std().tolist() #tolist()convert massive data into list format #Or the sign(+) concatinate both methods together
    features.append(label)
    return features

#empty list that append both dataset into one and concationation done by calling get_feature function
list=[]
list.append(get_features("/home/sonu-nitu/Downloads/gearboxdata (1)/healthy.csv/h30hz0.txt",0))
list.append(get_features("/home/sonu-nitu/Downloads/gearboxdata (1)/broken.csv/b30hz0.txt",1)) #healthy greabox=0,broken gaerbox=1
#final answer table for ai
finaldf=pd.DataFrame(list)
x=finaldf.iloc[:, :-1]# row,col,-1(dont give you last col becuse it contains answer )
y=finaldf.iloc[:,-1]#rw,col,-1(answer)


# Training the model 
model=RandomForestClassifier(random_state=42)
model.fit( x.to_numpy(), y.to_numpy() )
ypred=model.predict(x.to_numpy())
print("accuracy",accuracy_score(y.to_numpy(),ypred))    

import seaborn as sns
feature2=['mean_sens1','mean_sens2','mean_sens3','mean_sens4','std_sens1','std_sens2','std_sens3','std_sens4']
x.columns=feature2
sns.set_style("whitegrid")
plt.figure(figsize=(12,6))
ax=sns.scatterplot(x='mean_sens1',y='std_sens1',palette=['green','red'],hue=y,style=y,s=200,data=x)
plt.title("Mean vs Standard Deviation for Sensor 1")
plt.xlabel("Mean of Sensor 1")
plt.ylabel("Standard Deviation of Sensor 1")
handles, _ = ax.get_legend_handles_labels()
plt.legend(handles=handles, title="Gearbox Condition", labels=["Healthy (0)", "Broken (1)"])
plt.show()  