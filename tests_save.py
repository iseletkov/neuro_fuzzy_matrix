import NFuzMatrix # нейронная сеть
from NFuzMatrix import NFM # нейронная сеть
import os # работа с файлами
import numpy as np
import pickle  # сохрание и загрузка состояния нейросети 

# загрузка состояния сети
with open('NeuFuzMatrix_model.pkl', 'rb') as f:
    nfm_loaded = pickle.load(f)
# Текущая директория
current_dir = os.path.dirname(os.path.abspath(__file__))

# Путь к файлу относительно текущей директории
file_path = os.path.join(current_dir, "output.txt")
ts = np.loadtxt(file_path, usecols=[0,1,2])
X = ts[:,0:2]
Y = ts[:,2]
nfm = NFuzMatrix.NFM(X, Y)
nfm_loaded.train(epochs=20, k=0.00001)
print("Вычисленное: ", nfm_loaded.matrix_y)
print("Ожидаемое: ", nfm_loaded.Y)

# предсказание значений целевой переменной
X_test = np.array([[55, 3.5], [60, 0]])
y_test = NFM.predict(nfm_loaded, X_test)
print(y_test)