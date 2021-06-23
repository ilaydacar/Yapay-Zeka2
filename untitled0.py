# -*- coding: utf-8 -*-
"""
Created on Mon May 31 18:14:32 2021

@author: ilayda
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('odev_tenis.txt.crdownload')
#pd.read_csv("veriler.csv")
#test
print(veriler)


'''
#encoder: Kategorik -> Numeric dönüşüm yapalım.
play = veriler.iloc[:,-1:].values
print(play)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

play[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(play)


#sondan ikinci kolonu 1 ve 0 a dönüştürdük
windy = veriler.iloc[:,-2:-1].values
print(play)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

windy[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(windy)

#encoder: Kategorik -> Numeric
c = veriler.iloc[:,-1:].values
print(c)


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(c)

'''

#1)DÖNÜŞTÜRMEK

#yukrdaki gibi tek tek dönüştürmek yerine aşağıdaki tek kodla 1 ve 0 a dönüştürdük.
#sundy,rainy,overcast,windy,play labelenconder ile 1 ve 0 a dönüştürdük.
from sklearn.preprocessing import LabelEncoder
veriler2 = veriler.apply(LabelEncoder().fit_transform)


#temperature ve humidity onehot ettik. onehot ile yaptık çünkü zaten sayılardı.true falan değildi.
#temperature ve humidity 
c = veriler2.iloc[:,:1]
from sklearn import preprocessing
ohe = preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)


#2)EKLEME yapalım bir tabloda

#havadurumu ile dataframe yaparak rainy,sunny,overcast 1 ve 0 şeklinde bir tabloya ekledik.
havadurumu = pd.DataFrame(data = c, index = range(14), columns=['o','r','s'])

#sonveriler ile=havadurumuna veriler tablosundan windy ve play 0 ve 1 şeklini tabloya ekledik.
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)

#yukarda yazdırdığımız veriler2 temperature ve humidity de onehot şeklinde tabloya ekleyelim.
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler], axis = 1)




#3)VERİLERİ BÖLME
#humidity bağımlı değişken olduğu için o hariç hepsini bölüyoruz ayrı tabloda
#y_train ve y_test humadityi tabloda göstercek. sonveriler.iloc[:,-1:]
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#tahmin edelim. humidity tahmin edelim. y_test ile karşılaştırarak.
y_pred = regressor.predict(x_test)
print(y_pred)



#4)GERİYE ELEME
#Başarı ölçerek hangi verileri çıkarcağımıza bakalım. Önce tüm değişkenleri tanımlayalım.
#14 satır var. 
import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )

#6 kolon var. Kolon tanımlayalım. Sonrada sm.OLS ile tüm değişkenleri alsın.sonveriler.iloc[:,-1:]
X_l=sonveriler.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())



#4)VERİ ATMA
#yandaki raporda en yüksek olan p>t değeri x1 olduğu için 0.593 onu atıyoruz.
#2.playden sonrasını yazdırıp windy atıyoruz [:,1:]
sonveriler = sonveriler.iloc[:,1:]

#yeni tabloyu yazdıralım. x1 olmadığı yani windy olmadığı- kolon sayısı 5 oldu iloc[:,[0,1,2,3,4]]
import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )

X_l=sonveriler.iloc[:,[0,1,2,3,4]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())


#5)X_TRAİN VE X_TEST HUMİDİTY DEĞİŞKENİNİ ATMA

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]


#sonra y_test ve y_pred karşılaştırınca iyileşmiş halini görebilirsin.
regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)


#y_pred ilk tahmin 84 ken x1 i silip başarı oranını arttırınca 77 oldu ve y_testteki 70 e daha fazla yaklaştı.






