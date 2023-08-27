

#import csv
#from pickletools import optimize
#import numpy as np
#import matplotlib.pyplot as plt
#from tensorflow.keras.preprocessing.text import one_hot
#from sklearn.linear_model import Ridge, LinearRegression, Lasso
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Dense, Activation, Dropout, Input


#def prepare_data(list_len, data):

#   data = np.array(data)
   
#   assert data.shape == (list_len, 6)

#   max_len = 10 # число, до которого будет догоняться строка с нечисловыми столбцами(нужно тк у разных людей разная длинна профессии)
#   vocab_size = 500
#   new_data = np.zeros((list_len, 13), dtype=int)
   
#   # переделываем стобцы с 1го по 3й(включительно)
#   for i in range(len(data[:, 1:4])):
#       text = one_hot(' '.join(data[:, 1:4][i]), vocab_size)
#       text += ([0] * (max_len - len(text)))
#       new_data[:, 1:11][i] = text
   
#    # убираем столбцы с текстом для конвертации строк с числами к просто числами
#   data = np.delete(data, [1,2,3], axis=1)
   
#   for i in range(len(data)):
#       for j in range(len(data[i])):
#           data[i][j] = round (float(data[i][j]))
   
#   # Объединяем
#   new_data[:, 0] = data[:, 0]
#   new_data[:, -2:] = data[:, -2:]
   
#   x_data = new_data[:, :-1]
#   y_data = new_data[:, -1]
#   # переводим [n,n,n] к форме [[n], [n], [n]]
#   y_data = np.reshape(y_data, (list_len,1))

#   return x_data, y_data


#data = []

#with open('salary.csv') as file:
#    reader = csv.reader(file)
#    for i, line in enumerate(reader):
#        if i == 0:
#            continue
#        data.append(np.array(line))

#list_len = len(data)
#x_data, y_data = prepare_data(list_len=list_len, data=data)

#input_ = Input(12)
#out = Dense(256, activation='relu')(input_)
#out = Dropout(0.2)(out)
#out = Dense(128, activation='relu')(out)
#out = Dropout(0.2)(out)
#out = Dense(128, activation='relu')(out)
#out = Dense(1, activation='relu')(out)

#model = Model(input_, out)

#model.build((None, 12))
#model.compile(loss = 'mean_squared_logarithmic_error', optimizer='adam')

#model.fit(x_data, y_data, epochs = 200, batch_size=64, verbose=False)

## Составление массива для анализа
#to_predict = data[:15]

#for i in to_predict:
#    data = []
#    for j in range(10):
#        c = i.copy()
#        c[0] = str(float(c[0]) + (10 * j))
#        c[-2] = str(float(c[-2]) + (10 * j))
#        data.append(c)    
#    x, _ = prepare_data(len(data), data)
#    x = model.predict(x)
#    y = [ float(i) for i in np.array(data)[:, 4]]
#    plt.plot(y, x, label=data[0][3])
#plt.show()
##plt.plot(y_data, color='b', label='Answers')
##plt.plot(model.predict(x_data), color='r', label='Predicted')
##plt.show()


import csv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import one_hot
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor

def prepare_data(list_len, data):

   data = np.array(data)
   
   assert data.shape == (list_len, 6)

   max_len = 10 
   vocab_size = 500
   new_data = np.zeros((list_len, 13), dtype=int)
   

   for i in range(len(data[:, 1:4])):
       text = one_hot(' '.join(data[:, 1:4][i]), vocab_size)
       text += ([0] * (max_len - len(text)))
       new_data[:, 1:11][i] = text
   
   
   data = np.delete(data, [1,2,3], axis=1)
   
   for i in range(len(data)):
       for j in range(len(data[i])):
           data[i][j] = round (float(data[i][j]))
   

   new_data[:, 0] = data[:, 0]
   new_data[:, -2:] = data[:, -2:]
   
   x_data = new_data[:, :-1]
   y_data = new_data[:, -1]

   y_data = np.reshape(y_data, (list_len,1))

   return x_data, y_data


data = []

with open('salary.csv') as file:
    reader = csv.reader(file)
    for i, line in enumerate(reader):
        if i == 0:
            continue
        data.append(np.array(line))

list_len = len(data)
x_data, y_data = prepare_data(list_len=list_len, data=data)

model = DecisionTreeRegressor(random_state=1, max_depth=7)
model.fit(x_data, y_data)

to_predict = [
        ['20', 'Male', 'Bachelor\'s', 'Product Manager', '0', '0'],
        ['80', 'Female', 'Master\'s', 'Software Engineer', '0', '0'],
        ['45', 'Male', 'Bachelor\'s', 'Manager', '0', '0'],
        ['16', 'Female', 'PhD', 'IT Support', '0', '0']
    ]


#to_predict = data[:5]

for i in to_predict:
    data = []
    data.append(i)
    for j in range(10):
        c = i.copy()
        c[0] = str(float(c[0]) + (5 * j))
        c[-2] = str(float(c[-2]) + (5 * j))
        data.append(c)    
    x, _ = prepare_data(len(data), data)
    x = model.predict(x)
    y = [ float(i) for i in np.array(data)[:, 4]]
    plt.plot(y, x)

to_predict = np.array(to_predict)
arr = to_predict[:, 3][:len(to_predict)+1]
plt.legend(arr)
plt.show()
#plt.plot(y_data, color='b', label='Answers')
#plt.plot(model.predict(x_data), color='r', label='Predicted')
#plt.show()