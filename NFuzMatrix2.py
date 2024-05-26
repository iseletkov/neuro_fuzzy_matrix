import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import pickle  # сохрание и загрузка состояния нейросети 
import os # работа с файлами

# линия
def curve(bottom, top, x):
    return (x-bottom)/(top-bottom)
    
# трапеция
def Trapeze(bottom_left, top_left, top_right, bottom_right):
    def own(x):
        if (top_left<=x<=top_right): 
            return 1
        if (bottom_left<=x<top_left):
            return curve(bottom_left, top_left, x)
        if (top_right<x<=bottom_right):
            return curve(bottom_right, top_right,x)
        return 0
    return own

# треугольник
def Triangle(bottom_left, peak, bottom_right):
    def own(x):
        if (x==peak):
            return 1
        if (bottom_left<=x<peak):
            return curve(bottom_left, peak, x)
        if (peak<x<=bottom_right):
            return curve(bottom_right, peak, x)
        return 0
    return own

# Гауссовская
# a – координата максимума функции принадлежности; b – коэффициент концентрации функции принадлежности
def Gauss(a,b):
    def own(x):
        if (b > 0):
            return math.exp(-(x-a)**2/(2*b**2))  
        return 0
    return own

# Задание функции с помощью ломанных
def Points(list_point):
    x = np.array([d[0] for d in list_point])
    y = np.array([d[1] for d in list_point])
    def own(x_val):
        idx = np.searchsorted(x, x_val, side='right')
        if idx == 0:
            return y[0]
        if idx == len(x):
            return y[-1]
        x1, x2 = x[idx-1], x[idx]
        y1, y2 = y[idx-1], y[idx]
        return y1 + (x_val - x1) * (y2 - y1) / (x2 - x1)
    return own

# нечеткий вектор
class FuzzyVector():
    def __init__(self, positive):
        self.truth = positive
    
    def __str__(self):
        return f"truth: {round(self.truth, 2)}"

    def truth(self):
        return self.truth
    
    def inverse(self):
        return FuzzyVector(1-self.truth)

    def conjunction(self, other):
        positive = self.truth * other.truth
        return FuzzyVector(positive)

    def disjunction(self, other):
        positive = 1 - (1-self.truth) * (1-other.truth)
        return FuzzyVector(positive)

    def implication(self, other):
        return FuzzyVector(max(self.truth+other.truth-1, 0))
        
def conjunction(vectors):
    v = FuzzyVector(1)
    for vector in vectors:
        v = v.conjunction(vector)
    return v

def disjunction(vectors):
    v = FuzzyVector(0)
    for vector in vectors:
        v = v.disjunction(vector)
    return v

# лингвистическая переменная
class Feature():    
    def __init__(self, name, units, min, max, inout):
        self.name = name
        self.units = units
        self.min = min
        self.max = max
        self.predicates=[]
        self.linspace = None
        # Входной или рассчётный признак.
        self.inout = inout
        #Текущее значение для входных признаков
        self.value=None
        # Список правил, которые говорят о данном рассчётном признаке
        # чтобы много раз его не строить при вычислениях.
        self.rules = []

# термы ЛП
class FuzzyPredicate():
    def __init__(self, feature: Feature, name, func=None, params=None, const=None):
        self.feature: Feature = feature
        self.name = name
        #Для центроидного метода дефаззификации
        self.func = func
        self.params = params
        #Для упрощённого метода дефаззификации
        self.const = const

    def scalar(self,x=None):
        if x is None:
            if self.const is None:  
                raise ValueError(f"Const value for predicate {self.feature.name} '{self.name}' is not specified!") 
            else:
                return self.const

        if self.func is None:
            raise ValueError(f"Function for predicate {self.feature.name} '{self.name}' is not specified!")    

        f=self.func(self.params)
        return f(x)

    def vector(self,x=None):
        return FuzzyVector(self.scalar(x))
    
# правила 
class Rule():
    def __init__(self, input_pridicates, output_pridicate, weight):
        self.inputs = input_pridicates
        self.output = output_pridicate
        self.weight=weight
        self.truth = None

    def __str__(self):
        input_texts = [str(input.feature.name)+' "'+str(input.name)+'"' for input in self.inputs]
        text = "If "
        text += " and ".join(input_texts)
        text += ", then " + str(self.output.feature.name) + ' "' + str(self.output.name) +'". '
        text += "Truth: "+str(self.weight)
        return text
    

class Matrix():
    # агрегирование подусловий
    def __aggregation__(nfm):
        for rule in nfm.rules:
            inputs = rule.inputs
            rule.truth=conjunction([input.vector(input.feature.value) for input in inputs])

    # Активизация подзаключений
    def __activisation__(nfm):
        for rule in nfm.rules:
            rule.truth=rule.truth.implication(FuzzyVector(rule.weight))


    # Композиция и дефаззификация программно реализуются внутри одного цикла,
    # поэтому их надо писать внутри одной функции.
    def calculate(nfm):
        Matrix.__aggregation__(nfm)
        Matrix.__activisation__(nfm)

        # В общем виде, у алгоритма может быть несколько выходных переменных, для каждого выходного признака.
        # Поэтому на выходе должен быть массив.
        result = None
        for feature_out in nfm.features_out:

            rules = feature_out.rules
            if len(rules)==0:
                print(f"There is no rules for target feature: {feature_out.name}")
                result=np.nan
                continue

            numerator = 0
            denominator = 0


            # Метод дефазификации с помощью расчёта центра масс.
            if nfm.defuzzification=="Centroid":
                # Формируем набор значений из области значений выходного признака для расчёта интеграла.
                xarr = feature_out.linspace              
                for x in xarr:
                    y = (disjunction([rule.output.vector(x).conjunction(rule.truth) for rule in rules])).truth
                    numerator += x*y
                    denominator += y  
            # Упрощённый метод дефазификации 
            elif nfm.defuzzification=="Simple":
                for rule in rules:
                    numerator += (rule.output.vector().conjunction(rule.truth)).truth
                    # rule.truth - степень реализации правила в виде вектора, 
                    # rule.truth.truth - истинностная координата вектора степени реализации правила.
                    # print("rule.truth: ", rule.truth)
                    # print("rule.truth.truth: ", rule.truth.truth)
                    denominator += rule.truth.truth 

            # # Упрощённый метод дефазификации 
            # for rule in rules:
            #     numerator += (rule.output.vector().conjunction(rule.truth)).truth
            #     # rule.truth - степень реализации правила в виде вектора, 
            #     # rule.truth.truth - истинностная координата вектора степени реализации правила.
            #     denominator += rule.truth.truth 

            if denominator!=0:
                result=numerator/denominator
            else:
                #Ни одно из правил не выполнилось.
                result=np.nan

        return result

class NFM():
    def __init__(self, X, Y):
        self.X = np.array(copy.copy(X)) # X
        self.Y = np.array(copy.copy(Y)) # Y
        self.defuzzification= None #Centroid or Simple
        self.errors = [] # MSE
        self.predictedValues = [] # конечные обученные значения
        self.residuals = [] # конечная разница результатов обучения
        self.features_in = [] # входные лп 
        self.features_out = [] # выходные лп
        self.rules = [] # список правил
        self.num = 100 
        self.matrix_y = []

    def create_feature(self, name, units, min, max, inout):
        feature = Feature(name, units, min, max, inout)
        if inout:
            self.features_in.append(feature)
        else:
            feature.linspace = np.linspace(feature.min, feature.max, self.num)
            self.features_out.append(feature)
        return feature

    def create_predicate(self, feature: Feature, name, func=None, params = None, const=None):
        # Проверка, что признак принадлежит данной системе.
        if (feature in self.features_in or feature in self.features_out):
            predicate = FuzzyPredicate(feature, name, func, params, const)
            feature.predicates.append(predicate)
            return predicate
        else:
            raise Exception("The feature does not belong to this system.")

    def create_rule(self, input_predicates, output_predicate, weight):
        # Проверка, что предикаты принадлежат данной системе.
        for predicate in input_predicates:
            if not(predicate.feature in self.features_in):
                raise Exception("The pridicates does not belong to this system.")

        if not(output_predicate.feature in self.features_out):
            raise Exception("The pridicate does not belong to this system.")

        rule = Rule(input_predicates, output_predicate, weight)
        self.rules.append(rule)
        # Чтобы при вычислении значения признака сразу использовать только релевантные правила.
        output_predicate.feature.rules.append(rule)
        return rule

    def predict(self, x):
        # Проверка соответствия переданных значений количеству входных признаков.
        if (len(x[0,:]) != len(self.features_in)):
            raise Exception("Not matching the number of input parameters.")
        
        y=[]
        for row in range(len(x[:,0])):
            n=0
            for feature in self.features_in:
                if (feature.min<=x[row,:][n]<=feature.max):
                    feature.value=x[row,:][n]
                    n+=1
                else:
                    raise Exception(f"The value of the '{feature.name}' does not match the range.")
            y.append(Matrix.calculate(self))
        return y

    # обновление точек функции принадлежности или добавление новой
    def update_or_insert(self,params, x, dE_dP):
        for i, (f, s) in enumerate(params):
            if f == x:
                Yy=min(s+dE_dP, 1)
                y=max(Yy, 0)
                params[i] = (x, y)
                break
        else:
            f = Points(params)
            y = f(x)
            Yy=min(y+dE_dP, 1)
            y=max(y, 0)
            params.append((x, y))
        params.sort(key=lambda x: x[0])
        return params

    # обучение mini-batch learning, stochastic gradient descent, обучение по каждому множеству
    # количество эпох обучения, точность обучения, скорость обучения
    def train(self, epochs=5, tolerance=1e-1, k=0.001):
        # Проверка соответствия переданных значений количеству входных признаков.
        if (len(self.X[0,:]) != len(self.features_in)):
            raise Exception("Not matching the number of input parameters.")
        
        convergence = False
        epoch = 0
        while (epoch < epochs) and (convergence is not True):            
            self.matrix_y=[]
            # проход по каждому множеству
            for row in range(len(self.X[:,0])):
                n=0
                # прямой проход
                for feature in self.features_in:
                    if (feature.min<=self.X[row,:][n]<=feature.max):
                        feature.value=self.X[row,:][n]
                        n+=1
                    else:
                        raise Exception(f"The value of the '{feature.name}' does not match the range.")
                        
                predicted=Matrix.calculate(self)
                self.matrix_y.append(predicted)
                # обратный проход и обновление
                error=predicted-self.Y[row] 
                # if predicted is not np.nan:
                for rule in self.rules:
                    inputs = rule.inputs
                    out=rule.output
                    error=error/(out.feature.max-out.feature.min)
                    for input in inputs:
                        # значение смещения
                        dE_dP=k*error*rule.truth.truth*input.vector(input.feature.value).truth
                        # обновление графика
                        params = input.params
                        params=self.update_or_insert(params, input.feature.value, dE_dP)    
                        input.params = params

            epoch+=1
            # функция потерь MSE
            # errors = np.sum((np.array(self.Y)-np.array(self.matrix_y))**2)/len(self.Y)
            # функция потерь RMSE
            errors = np.sqrt(np.sum((np.array(self.Y)-np.array(self.matrix_y))**2)/len(self.Y))
            # ошибка предсказания
            residuals = np.array(self.Y)-np.array(self.matrix_y)
            self.errors.append(errors)
            # проверка точности обучения 
            if errors < tolerance:
                convergence = True

            # изменение скорости обучения
            if len(self.errors) >= 5:
                if (self.errors[-5] > self.errors[-4] > self.errors[-3] > self.errors[-2] > self.errors[-1]):
                    k = k * 1.1

            if len(self.errors) >= 5:
                if (self.errors[-1] < self.errors[-2]) and (self.errors[-3] < self.errors[-2]) and (self.errors[-3] < self.errors[-4]) and (self.errors[-5] > self.errors[-4]):
                    k = k * 0.9
                

    # Графики принадлежности термов ЛП 

    # def show_view(self):
    #     lp=self.features_in
    #     for feature in lp:
    #         x=np.linspace(feature.min, feature.max, self.num)
    #         fig, ax = plt.subplots(1, len(feature.predicates))
    #         n=0
    #         for predicate in feature.predicates:
    #             f = predicate.params
    #             x = [d[0] for d in f]
    #             y = [d[1] for d in f]
    #             ax[n].set(xlabel=feature.units, ylabel = "Степень принадлежности")
    #             ax[n].set_title(feature.name+f" '{predicate.name}'")
    #             fig.set_figwidth(8)
    #             fig.set_figheight(3)
    #             ax[n].plot(x, y, clip_on = False)
    #             plt.tight_layout()
    #             n+=1
    #         plt.show()

    def show_view(self):
        lp=self.features_in
        for feature in lp:
            x=np.linspace(feature.min, feature.max, self.num)
            fig, ax = plt.subplots(1, len(feature.predicates))
            n=0
            for predicate in feature.predicates:
                y=[]
                for xx in x:
                    f = predicate.func(predicate.params)
                    y.append(f(xx))
                ax[n].set(xlabel=feature.units, ylabel = "Степень принадлежности")
                ax[n].set_title(feature.name+f" '{predicate.name}'")
                fig.set_figwidth(8)
                fig.set_figheight(3)
                ax[n].plot(x, y, clip_on = False)
                plt.tight_layout()
                n+=1
            plt.show()





    # График метрики RSME или MSE
    def show_errors(self):
        plt.plot(self.errors)
        plt.xlabel('Эпоха')
        plt.ylabel('Ошибка')
        plt.show()
