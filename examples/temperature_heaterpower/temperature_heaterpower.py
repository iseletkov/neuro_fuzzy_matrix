import os
import sys
# Добавляем путь к директории проекта в sys.path
project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_directory)
from NFuzMatrix import *

# Пример использования кода:
x = [[55], [50]]
y = [60, 55]

nfm = NFM(x, y)
nfm.defuzzification = "Simple"
f_temp = nfm.create_feature("Температура", "C", 0, 100, True)
power = nfm.create_feature("Мощность нагревателя", "Вт", 0, 100, False)
p_temp_low = nfm.create_predicate(f_temp, 'низкая', func = Points, params=[(0, 0), (0, 1), (100, 0)])
p_temp_high = nfm.create_predicate(f_temp, 'высокая', func = Points, params = [(0, 0), (100,1), (100,0)])

power_low = nfm.create_predicate(power, 'низкое', const=0)
power_high = nfm.create_predicate(power, 'высокое', const=100)

r_1 = nfm.create_rule([p_temp_low], power_high, 1)
r_2 = nfm.create_rule([p_temp_high], power_low, 1)

nfm.show_view()
print("Вычисленное: ", nfm.matrix_y)
print("Ожидаемое: ", nfm.Y)

nfm.train(epochs=10, k=1)
print("Вычисленное: ", nfm.matrix_y)
print("Ожидаемое: ", nfm.Y)

print("errors: ", nfm.errors)

nfm.show_view(True)
nfm.show_errors(True)

# сохранения состояния нейросети
# with open('NeuFuzMatrix_model.pkl', 'wb') as f:
#     pickle.dump(nfm, f)



