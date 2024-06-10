import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import logging
from tqdm import trange


# линия
def curve(bottom, top, x):
    return (x - bottom) / (top - bottom)


# трапеция
def trapeze(bottom_left, top_left, top_right, bottom_right):
    def own(x):
        if (top_left <= x <= top_right):
            return 1
        if (bottom_left <= x < top_left):
            return curve(bottom_left, top_left, x)
        if (top_right < x <= bottom_right):
            return curve(bottom_right, top_right, x)
        return 0

    return own


# треугольник
def triangle(bottom_left, peak, bottom_right):
    def own(x):
        if x == peak:
            return 1
        if bottom_left <= x < peak:
            return curve(bottom_left, peak, x)
        if peak < x <= bottom_right:
            return curve(bottom_right, peak, x)
        return 0

    return own


# Гауссовская
# a – координата максимума функции принадлежности; b – коэффициент концентрации функции принадлежности
def gauss(a, b):
    def own(x):
        if b > 0:
            return math.exp(-(x - a) ** 2 / (2 * b ** 2))
        return 0

    return own


# Кусочно-линейная функция
def piecewise(xarr, **kwargs):
    yarr = []
    points = kwargs["points"]
    x1 = points[0][0]
    x2 = points[1][0]
    idx = 1
    for x_val in xarr:
        if x_val < x1:
            yarr.append(0)
            continue
        while x_val > x2 and idx < len(points)-1:
            idx += 1
            x1 = x2
            x2 = points[idx][0]

        if x_val > x2:
            k = len(yarr)
            for i in range(len(xarr)-k):
                yarr.append(0)
            break

        if x_val == x1:
            yarr.append(points[idx - 1][1])
            continue

        y1, y2 = points[idx - 1][1], points[idx][1]
        yarr.append(y1 + (x_val - x1) * (y2 - y1) / (x2 - x1))
    return yarr


# нечеткий вектор
class FuzzyVector:
    def __init__(self, positive):
        self.truth = positive

    def __str__(self):
        return f"truth: {round(self.truth, 2)}"

    def truth(self):
        return self.truth

    def inverse(self):
        return FuzzyVector(1 - self.truth)

    def conjunction(self, other):
        positive = self.truth * other.truth
        return FuzzyVector(positive)

    def disjunction(self, other):
        positive = 1 - (1 - self.truth) * (1 - other.truth)
        return FuzzyVector(positive)

    def implication(self, other):
        return FuzzyVector(max(self.truth + other.truth - 1, 0))


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
class Feature:
    def __init__(self, name, units, min, max, inout):
        self.name = name
        self.units = units
        self.min = min
        self.max = max
        self.size = max-min
        self.predicates = []
        self.linspace = np.linspace(self.min, self.max, 200)
        # Входной или рассчётный признак.
        self.inout = inout
        # Текущее значение для входных признаков
        self.value = None
        # Список правил, которые говорят о данном рассчётном признаке
        # чтобы много раз его не строить при вычислениях.
        self.rules = []


# термы ЛП
class FuzzyPredicate:
    def __init__(self, feature: Feature, name, const=None, func=None, **kwargs):
        self.feature: Feature = feature
        self.name = name
        self.centre = None
        # Для упрощённого метода дефаззификации
        self.const = const
        if const is None:
            self.__init_points__(func, **kwargs)

    def __init_points__(self, func, **kwargs):
        self.yarr = func(self.feature.linspace, **kwargs)

    def __get_value__(self, x_val):
        points_x = self.feature.linspace
        idx = np.searchsorted(points_x, x_val, side='right')
        if idx == 0:
            return self.yarr[0]
        if idx == len(points_x):
            return self.yarr[-1]
        x1, x2 = points_x[idx - 1], points_x[idx]
        y1, y2 = self.yarr[idx - 1], self.yarr[idx]
        return y1 + (x_val - x1) * (y2 - y1) / (x2 - x1)

    # обновление точек функции истинности
    def update_func(self, x_val, dedp):
        points_x = self.feature.linspace

        for idx in range(len(points_x)-1):
            if points_x[idx] <= x_val < points_x[idx + 1]:
                if idx > 1:
                    yy = min(self.yarr[idx-1] + dedp*0.5, 1)
                    y = max(yy, 0)
                    self.yarr[idx-1] = y

                yy = min(self.yarr[idx] + dedp, 1)
                y = max(yy, 0)
                self.yarr[idx] = y

                yy = min(self.yarr[idx+1] + dedp, 1)
                y = max(yy, 0)
                self.yarr[idx+1] = y

                if idx < len(points_x)-1:
                    yy = min(self.yarr[idx+2] + dedp*0.5, 1)
                    y = max(yy, 0)
                    self.yarr[idx+2] = y

                break
        return

    def scalar(self, x=None):
        if x is None:
            if self.const is None:
                raise ValueError(f"Const value for predicate {self.feature.name} '{self.name}' is not specified!")
            else:
                return self.const

        if self.yarr is None:
            raise ValueError(f"Function for predicate {self.feature.name} '{self.name}' is not specified!")

        return self.__get_value__(x)

    def vector(self, x=None):
        return FuzzyVector(self.scalar(x))


# правила 
class Rule:
    def __init__(self, input_pridicates, output_pridicate, weight):
        self.inputs = input_pridicates
        self.output = output_pridicate
        self.weight = weight
        self.truth = None

    def __str__(self):
        input_texts = [str(input.feature.name) + ' "' + str(input.name) + '"' for input in self.inputs]
        text = "If "
        text += " and ".join(input_texts)
        text += ", then " + str(self.output.feature.name) + ' "' + str(self.output.name) + '". '
        text += "Truth: " + str(self.weight)
        return text


class NFM:
    def __init__(self, X, Y, level=logging.INFO):
        self.X = np.array(copy.copy(X))  # X
        self.Y = np.array(copy.copy(Y))  # Y
        self.defuzzification = None  # Centroid or Simple
        self.errors = []  # RMSE
        self.residuals = []  # конечная разница результатов обучения
        self.features_in = []  # входные лп
        self.features_out = []  # выходные лп
        self.rules = []  # список правил
        self.num = 200
        self.matrix_y = []

        self.logger = logging.getLogger("NFM")
        self.logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        self.logger.addHandler(ch)

    # агрегирование подусловий
    def __aggregation__(self):
        for rule in self.rules:
            inputs = rule.inputs
            rule.truth = conjunction([input.vector(input.feature.value) for input in inputs])

    # Активизация подзаключений
    def __activisation__(self):
        for rule in self.rules:
            rule.truth = rule.truth.implication(FuzzyVector(rule.weight))

    # Композиция и дефаззификация программно реализуются внутри одного цикла,
    # поэтому их надо писать внутри одной функции.
    def calculate(self):
        self.__aggregation__()
        self.__activisation__()

        # В общем виде, у алгоритма может быть несколько выходных переменных, для каждого выходного признака.
        # Поэтому на выходе должен быть массив.
        result = None
        for feature_out in self.features_out:

            rules = feature_out.rules
            if len(rules) == 0:
                # print(f"There is no rules for target feature: {feature_out.name}")
                self.logger.error(
                    "There is no rules for target feature: {feature_out.name}"
                )
                result = np.nan
                continue

            numerator = 0
            denominator = 0

            # Метод дефазификации с помощью расчёта центра масс.
            if self.defuzzification == "Centroid":
                # Формируем набор значений из области значений выходного признака для расчёта интеграла.
                xarr = feature_out.linspace
                for x in xarr:
                    y = (disjunction([rule.output.vector(x).conjunction(rule.truth) for rule in rules])).truth
                    numerator += x * y
                    denominator += y
                    # Упрощённый метод дефазификации
            elif self.defuzzification == "Simple":
                for rule in rules:
                    numerator += (rule.output.vector().conjunction(rule.truth)).truth
                    # rule.truth - степень реализации правила в виде вектора,
                    # rule.truth.truth - истинностная координата вектора степени реализации правила.
                    # print("rule.truth: ", rule.truth)
                    # print("rule.truth.truth: ", rule.truth.truth)
                    denominator += rule.truth.truth

            if denominator != 0:
                result = numerator / denominator
            else:
                rules[0].inputs[0].vector(8.51)

                # Ни одно из правил не выполнилось.
                self.logger.debug("No rule found: " + (
                    ', '.join(f"{feature.name}: {feature.value}" for feature in self.features_in)))
                result = np.nan

        return result

    def create_feature(self, name, units, min, max, inout):
        feature = Feature(name, units, min, max, inout)
        if inout:
            self.features_in.append(feature)
        else:
            self.features_out.append(feature)
        return feature

    def create_predicate(self, feature: Feature, name, const=None, func=None, **kwargs):
        # Проверка, что признак принадлежит данной системе.
        if feature in self.features_in or feature in self.features_out:
            predicate = FuzzyPredicate(feature, name, const, func, **kwargs)
            feature.predicates.append(predicate)
            return predicate
        else:
            raise Exception("The feature does not belong to this system.")

    def create_rule(self, input_predicates, output_predicate, weight):
        # Проверка, что предикаты принадлежат данной системе.
        for predicate in input_predicates:
            if not (predicate.feature in self.features_in):
                raise Exception("The pridicates does not belong to this system.")

        if not (output_predicate.feature in self.features_out):
            raise Exception("The pridicate does not belong to this system.")

        rule = Rule(input_predicates, output_predicate, weight)
        self.rules.append(rule)
        # Чтобы при вычислении значения признака сразу использовать только релевантные правила.
        output_predicate.feature.rules.append(rule)
        return rule

    def predict(self, x):
        # Проверка соответствия переданных значений количеству входных признаков.
        if len(x[0, :]) != len(self.features_in):
            raise Exception("Not matching the number of input parameters.")

        y = []
        for row in range(len(x[:, 0])):
            n = 0
            for feature in self.features_in:
                if feature.min <= x[row, :][n] <= feature.max:
                    feature.value = x[row, :][n]
                    n += 1
                else:
                    raise Exception(f"The value of the '{feature.name}' does not match the range.")
            y.append(self.calculate())
        return y

    def centre_mass_out(self):
        for features in self.features_out:
            for predicates in features.predicates:
                if predicates.const is None:
                    xarr = features.linspace
                    numerator = 0
                    denominator = 0
                    for x in xarr:
                        y = predicates.scalar(x)
                        numerator += x * y
                        denominator += y
                    predicates.centre = numerator / denominator
                    # print(features.name, predicates.name, predicates.centre)

    # количество эпох обучения, точность обучения, скорость обучения
    def train(self, epochs=5, tolerance=1e-1, k=0.001):
        # Проверка соответствия переданных значений количеству входных признаков.
        if len(self.X[0, :]) != len(self.features_in):
            raise Exception("Not matching the number of input parameters.")

        convergence = False
        epoch = 0
        n = 0
        self.centre_mass_out()

        # Расчёт исходного значения метрики до обучения.
        matrix_y = []
        for row in range(len(self.X[:, 0])):
            n = 0
            # прямой проход
            for feature in self.features_in:
                if feature.min <= self.X[row, :][n] <= feature.max:
                    feature.value = self.X[row, :][n]
                    n += 1
                else:
                    raise Exception(f"The value of the '{feature.name}' does not match the range.")
            if self.features_in[0].value==8.57:
                xxx = 0
            predicted = self.calculate()
            matrix_y.append(predicted)
        # функция потерь RMSE
        errors = np.sqrt(np.sum((np.array(self.Y) - np.array(matrix_y)) ** 2) / len(self.Y))

        while (epoch < epochs) and (convergence is not True):
            self.matrix_y = []
            pbar = trange(len(self.X[:, 0]))
            pbar.set_description(f"Epoch {epoch+1: >5} ")
            pbar.set_postfix_str(f"RMSE: {round(errors, 5)}")
            # проход по каждому множеству
            for row in pbar:
                n = 0
                # прямой проход
                for feature in self.features_in:
                    if feature.min <= self.X[row, :][n] <= feature.max:
                        feature.value = self.X[row, :][n]
                        n += 1
                    else:
                        raise Exception(f"The value of the '{feature.name}' does not match the range.")

                predicted = self.calculate()
                self.matrix_y.append(predicted)

                if 10.55 <= self.features_in[0].value <= 10.87:
                    xxx=1
                # обратный проход и обновление
                error = predicted - self.Y[row]
                for rule in self.rules:
                    inputs = rule.inputs
                    out = rule.output

                    error1 = error / (out.feature.size**2)

                    if out.const is None:
                        error1 *= (out.feature.size - (out.centre - predicted))
                    else:
                        error1 *= (out.feature.size - (out.const - predicted))

                    for input in inputs:
                        # значение смещения
                        dedp = k * error1 * input.vector(input.feature.value).truth
                        # * rule.truth.truth

                        # обновление графика
                        input.update_func(input.feature.value, dedp)

            epoch += 1
            # функция потерь MSE
            # errors = np.sum((np.array(self.Y)-np.array(self.matrix_y))**2)/len(self.Y)
            # функция потерь RMSE
            errors = np.sqrt(np.sum((np.array(self.Y) - np.array(self.matrix_y)) ** 2) / len(self.Y))

            # ошибка предсказания
            self.residuals = np.array(self.Y) - np.array(self.matrix_y)
            self.errors.append(errors)
            # проверка точности обучения 
            if errors < tolerance:
                convergence = True

            # изменение скорости обучения
            if len(self.errors) >= 5:
                if (self.errors[-5] > self.errors[-4] > self.errors[-3] > self.errors[-2] > self.errors[-1]):
                    k = k * 1.1

            if len(self.errors) >= 5:
                if (self.errors[-1] < self.errors[-2]) and (self.errors[-3] < self.errors[-2]) and (
                        self.errors[-3] < self.errors[-4]) and (self.errors[-5] > self.errors[-4]):
                    k = k * 0.9

            self.show_view(True)

    # Графики принадлежности термов входных ЛП
    def show_view(self, block=False):
        lp = self.features_in

        flagMultiple = len(lp) > 1
        fig, ax = plt.subplots(len(lp))
        fig.set_figwidth(10)
        fig.set_figheight(5)
        n = 0
        for feature in lp:
            if flagMultiple:
                axx = ax[n]
            else:
                axx = ax
            axx.set(xlabel=feature.units, ylabel="Степень истинности, доли от полной истинности")
            axx.set_title(feature.name)
            x = feature.linspace
            for predicate in feature.predicates:
                y = predicate.yarr
                axx.plot(x, y, clip_on=False)
            n += 1
        plt.tight_layout()
        plt.show(block=block)

    # График метрики RSME или MSE
    def show_errors(self, block=False):
        plt.plot(self.errors)
        plt.xlabel('Эпоха')
        plt.ylabel('Ошибка')
        plt.show(block=block)
