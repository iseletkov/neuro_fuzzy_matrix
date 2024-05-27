import NFuzMatrix # нейронная сеть
from NFuzMatrix import Points

# Пример использования кода:
x = [[55], [50]]
y = [60, 55]

nfm = NFuzMatrix.NFM(x, y)
nfm.defuzzification = "Simple"
# nfm.defuzzification = "Centroid"
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


# errors = np.array(nfm.Y)-np.array(nfm.matrix_y).flatten()
# print("errors: ", errors)
print("errors: ", nfm.errors)

# pressure = nfm.predict(X)
# print(f"Значения давления: {pressure}")

# pressure = nfm.predict(np.array([[84, 7], [30, 4.8], [28, 2.2]]))  #85.06422, 78.0, 27.17808
# print(f"Значения давления: {pressure}")

# # print(nfm.X)

nfm.show_view(True)
nfm.show_errors(True)

# сохранения состояния нейросети
# with open('NeuFuzMatrix_model.pkl', 'wb') as f:
#     pickle.dump(nfm, f)



