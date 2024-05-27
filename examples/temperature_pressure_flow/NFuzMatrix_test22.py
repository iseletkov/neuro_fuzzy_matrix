import NFuzMatrix2 # нейронная сеть
from NFuzMatrix2 import Points
import os # работа с файлами
import numpy as np
import pickle  # сохрание и загрузка состояния нейросети 

# Пример использования кода:
# Текущая директория
current_dir = os.path.dirname(os.path.abspath(__file__))

# Путь к файлу относительно текущей директории
file_path = os.path.join(current_dir, "output.txt")

ts = np.loadtxt(file_path, usecols=[0,1,2])
X = ts[:,0:2]
Y = ts[:,2]
# X=np.array([[70, 3.8]])
# Y=[69.64384]

nfm = NFuzMatrix2.NFM(X, Y)
nfm.defuzzification = "Simple"
# nfm.defuzzification = "Centroid"
f_temp = nfm.create_feature("Температура", "C", 0, 150, True)
f_flow = nfm.create_feature("Расход", "м3/ч", 0, 8, True)
f_pressure = nfm.create_feature("Давление", "МПа", -100, 200, False)
p_temp_low = nfm.create_predicate(f_temp, 'низкая', func = Points, params = [(0,0), (0,1), (50,1), (100,0)])
p_temp_normal = nfm.create_predicate(f_temp, 'средняя', func = Points, params = [(0,0), (50,1), (100,1),(150,0)])
p_temp_high = nfm.create_predicate(f_temp, 'высокая', func = Points, params = [(50,0), (100,1), (150,1), (150,0)])
p_flow_low = nfm.create_predicate(f_flow, 'малый', func = Points, params = [(0,0), (2,1), (4,0)])
p_flow_normal = nfm.create_predicate(f_flow, 'средний', func = Points, params = [(2,0), (4,1), (6,0)])
p_flow_high = nfm.create_predicate(f_flow, 'большой', func = Points, params = [(4,0), (6,1), (8,0)])
p_pressure_low = nfm.create_predicate(f_pressure, 'низкое', const=0)
p_pressure_normal = nfm.create_predicate(f_pressure, 'среднее', const=50)
p_pressure_high = nfm.create_predicate(f_pressure, 'высокое', const=100)
# p_pressure_low = nfm.create_predicate(f_pressure, 'низкое', func = Points, params = [(-100,0), (0,1), (100,0)])
# p_pressure_normal = nfm.create_predicate(f_pressure, 'среднее', func = Points, params = [(0,0), (50,1), (100,0)])
# p_pressure_high = nfm.create_predicate(f_pressure, 'высокое', func = Points, params = [(0,0), (100,1), (200,0)])
r_1 = nfm.create_rule([p_temp_low, p_flow_low], p_pressure_low, 1)
r_2 = nfm.create_rule([p_temp_normal], p_pressure_normal, 1)
r_3 = nfm.create_rule([p_temp_high], p_pressure_high, 1)
r_4 = nfm.create_rule([p_flow_high], p_pressure_high, 1)

# nfm.show_view()
nfm.train(epochs=135, k=0.0001)
print("Вычисленное: ", nfm.matrix_y)
print("Ожидаемое: ", nfm.Y)


# errors = np.array(nfm.Y)-np.array(nfm.matrix_y).flatten()
# print("errors: ", errors)
print("errors: ", nfm.errors)

# pressure = nfm.predict(X)
# print(f"Значения давления: {pressure}")

pressure = nfm.predict(np.array([[84, 7], [30, 4.8], [28, 2.2]]))  #85.06422, 78.0, 27.17808
print(f"Значения давления: {pressure}")

# # print(nfm.X)

nfm.show_view()
nfm.show_errors()

# сохранения состояния нейросети
# with open('NeuFuzMatrix_model.pkl', 'wb') as f:
#     pickle.dump(nfm, f)



