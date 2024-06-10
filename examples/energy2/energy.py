from NFuzMatrix import *
import pandas as pd

df_train = pd.read_csv("Train.csv")
# df_train = df_train.sample(n=500, random_state=41)
# df_train = df_train[(19.9 <= df_train["AT"]) &( df_train["AT"]<= 20)]
x = df_train[["AT"]]
y = df_train["PE"]

nfm = NFM(x, y, level=logging.DEBUG)
nfm.defuzzification = "Simple"

AT = nfm.create_feature("Температура", "C", 0, 38, True)
# V = nfm.create_feature("Вытяжной вакуум", "cm Hg", 25, 82, True)
PE = nfm.create_feature("Генерация энергии", "MW", 430, 510, False)

p_AT_low = nfm.create_predicate(AT, 'низкая', func=piecewise, points=[[0, 0], [0, 1], [10, 1], [15, 0]])
p_AT_low1 = nfm.create_predicate(AT, 'низкая1', func=piecewise, points=[[10, 0], [15, 1], [20, 0]])
p_AT_normal = nfm.create_predicate(AT, 'средняя', func=piecewise, points=[[15, 0], [20, 1], [25, 1], [35, 0]])
p_AT_high = nfm.create_predicate(AT, 'высокая', func=piecewise, points=[[25, 0], [35, 1], [38, 1], [38, 0]])

# p_V_low = nfm.create_predicate(V, 'малый', func = Points, params = [(25,0), (25,1), (40,1), (50,0)])
# p_V_normal = nfm.create_predicate(V, 'средний', func = Points, params = [(40,0), (50,1), (60,0)])
# p_V_normal1 = nfm.create_predicate(V, 'средний1', func = Points, params = [(50,0), (60,1), (70,0)])
# p_V_high = nfm.create_predicate(V, 'большой', func = Points, params = [(60,0), (70,1), (82,1), (82,0)])

p_PE_low = nfm.create_predicate(PE, 'низкая', const=430)
p_PE_450 = nfm.create_predicate(PE, 'средняя', const=450)
p_PE_460 = nfm.create_predicate(PE, 'средняя', const=460)
p_PE_470 = nfm.create_predicate(PE, 'средняя', const=470)
p_PE_480 = nfm.create_predicate(PE, 'средняя', const=480)
p_PE_high = nfm.create_predicate(PE, 'высокая', const=510)

r_1 = nfm.create_rule([p_AT_low], p_PE_high, 1)
r_2 = nfm.create_rule([p_AT_low1], p_PE_480, 1)
r_3 = nfm.create_rule([p_AT_normal], p_PE_460, 1)
r_4 = nfm.create_rule([p_AT_high], p_PE_low, 1)

# r_1 = nfm.create_rule([p_AT_low, p_V_low], p_PE_high, 1)
# r_2 = nfm.create_rule([p_AT_high, p_V_high], p_PE_low, 1)
# r_3 = nfm.create_rule([p_AT_normal, p_V_high], p_PE_450, 1)
# r_4 = nfm.create_rule([p_AT_normal, p_V_normal1], p_PE_460, 1)
# r_5 = nfm.create_rule([p_AT_normal, p_V_normal], p_PE_470, 1)
# r_6 = nfm.create_rule([p_AT_low1, p_V_low], p_PE_480, 1)


nfm.show_view(True)
nfm.train(epochs=10, k=0.1)
# df_predict = pd.DataFrame({'Ожидаемое': nfm.Y, 'Вычисленное': nfm.matrix_y})
# print(df_predict)
nfm.show_view(True)
# print("errors: ", nfm.errors)
