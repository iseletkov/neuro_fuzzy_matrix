import os
import sys
# Добавляем путь к директории проекта в sys.path
project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_directory)
from NFuzMatrix import *
import numpy as np
import pickle  # сохрание и загрузка состояния нейросети

# Пример использования кода:
# Текущая директория
current_dir = os.path.dirname(os.path.abspath(__file__))

# Путь к файлу относительно текущей директории
file_path = os.path.join(current_dir, "output1.txt")

ts = np.loadtxt(file_path, usecols=[0,1,2,3,4])
X = ts[:,0:4]
Y = ts[:,4]
# X=np.array([[2, 30, 999, 30]])
# Y=[469.64384]

nfm = NFM(X, Y)
nfm.defuzzification = "Simple"
# nfm.defuzzification = "Centroid"
AT = nfm.create_feature("Температура", "C", 0, 38, True)
V = nfm.create_feature("Вытяжной вакуум", "cm Hg", 25, 82, True)
AP = nfm.create_feature("Давление", "milibar", 980, 1040, True)
RH = nfm.create_feature("Влажность воздуха", "%", 1, 101, True)
PE = nfm.create_feature("Электрическая энергия", "MW", 430, 510, False)

p_AT_low = nfm.create_predicate(AT, 'низкая', func = Points, params = [(0,0), (10,1), (20,0)])
p_AT_normal = nfm.create_predicate(AT, 'средняя', func = Points, params = [(15,0), (20,1), (35,0)])
p_AT_high = nfm.create_predicate(AT, 'высокая', func = Points, params = [(20,0), (30,1), (38,0)])

p_V_low = nfm.create_predicate(V, 'малый', func = Points, params = [(25,0), (35,1), (45,0)])
p_V_normal = nfm.create_predicate(V, 'средний', func = Points, params = [(30,0), (65,1), (70,0)])
p_V_high = nfm.create_predicate(V, 'большой', func = Points, params = [(60,0), (82,1), (82,0)])

p_AP_low = nfm.create_predicate(AP, 'малый', func = Points, params = [(990,0), (990,1), (1000,0)])
p_AP_normal = nfm.create_predicate(AP, 'средний', func = Points, params = [(993,0), (1018,1), (1030,0)])
p_AP_high = nfm.create_predicate(AP, 'большой', func = Points, params = [(1000,0), (1020,1), (1040,1), (1040,0)])

p_RH_low = nfm.create_predicate(RH, 'малый', func = Points, params = [(0,0), (20,1), (40,0)])
p_RH_normal = nfm.create_predicate(RH, 'средний', func = Points, params = [(30,0), (50,1), (70,0)])
p_RH_high = nfm.create_predicate(RH, 'большой', func = Points, params = [(60,0), (80,1), (100,0)])

p_PE_low = nfm.create_predicate(PE, 'низкое', const=440)
p_PE_normal = nfm.create_predicate(PE, 'среднее', const=470)
p_PE_high = nfm.create_predicate(PE, 'высокое', const=500)

# r_1 = nfm.create_rule([p_AT_low, p_AP_low, p_V_low, p_RH_low], p_PE_high, 1)
# r_2 = nfm.create_rule([p_AT_low, p_AP_low, p_RH_normal, p_V_high], p_PE_normal, 1)
# r_3 = nfm.create_rule([p_AT_low, p_AP_high], p_PE_low, 1)
# r_4 = nfm.create_rule([p_AP_high, p_V_low], p_PE_high, 1)
# r_5 = nfm.create_rule([p_AT_high, p_RH_high], p_PE_low, 1)


r_1 = nfm.create_rule([p_AT_low, p_V_low], p_PE_high, 1)
r_2 = nfm.create_rule([p_AT_normal, p_AP_low], p_PE_normal, 1)
r_3 = nfm.create_rule([p_AT_high, p_V_high], p_PE_low, 1)
r_4 = nfm.create_rule([p_RH_low, p_V_high], p_PE_high, 1)
r_5 = nfm.create_rule([p_RH_high, p_AP_high], p_PE_low, 1)
r_6 = nfm.create_rule([p_RH_high, p_AP_normal], p_PE_normal, 1)


# nfm.show_view()
nfm.train(epochs=145, k=0.00001)
print("Вычисленное: ", nfm.matrix_y)
print("Ожидаемое: ", nfm.Y)

print("errors: ", nfm.errors)


file_path = os.path.join(current_dir, "test.txt")
ts = np.loadtxt(file_path, usecols=[0,1,2,3,4])
XX = ts[:,0:4]
energy = nfm.predict(XX)
print(f"Выход электрической энергии: {energy}")

# energy = nfm.predict(np.array([[23.0, 66.05, 1020.61, 80.29], [26.84, 70.36, 1007.41, 71.47], [7.09, 43.13, 1017.91, 92.37]]))  #434.641093, 466.972021393, 497.1699038
# print(f"Выход электрической энергии: {energy}")


nfm.show_view(True)
# nfm.show_errors(True)

# сохранения состояния нейросети
# with open('NeuFuzMatrix_model.pkl', 'wb') as f:
#     pickle.dump(nfm, f)



# Коэффициент детерминации (R²)
y_mean = np.mean(Y)
ss_res = np.sum((Y - nfm.matrix_y) ** 2)
ss_tot = np.sum((Y - y_mean) ** 2)
r2 = 1 - (ss_res / ss_tot)
print("r2: ", r2)