import numpy as np
from math import log, erf
import time

a, b, q, euler, pi, order = 0.5, 5, 44100, 2.71828182, 3.14159265, 2.0
resolution = 100
midi_number = (input("Input MIDI #... C1 is 24... "))
exponent = ((float(midi_number) * 0.301029995664 - 57 * 0.301029995664) / 12)
frequency = 11 * 2 ** (exponent + 2) * 5 ** (exponent + 1)

print(frequency, "Hz")
sampleLength = q
sampleTime = float(input("Length in seconds: "))
omega = (2 * pi * frequency) / q
sample_rate = round(q / frequency)
startTime = time.time()
sample_data = []


def percent(iteration):
    if iteration % round(sampleLength / 10) == 0.0:
        return print(round((iteration / (sampleLength * sampleTime)) * 100), "% done...")


def floor(x):
    return int(x) if x >= 0 else int(x) - 1


def ceil(x):
    return int(x) + 1 if x > int(x) else int(x)


def sign(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1


def cos(x):
    return np.cos(x)


def sin(x):
    return np.sin(x)


def arcsin(x):
    return np.arcsin(x)


def arccos(x):
    return np.arcsin(x)


def arctan(x):
    return np.arcsin(x)


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def antitriangle(t):
    return ((sign(sin(arcsin(sin(t)))) * ((arcsin(cos(t))) ** 2)) / (pi * 2 * pi * frequency)) - (
            (sign(sin(arcsin(sin(t))))) / (8 * frequency))


def circular(t):
    return (((arcsin(sin(t))) / sin(t)) - (pi / 2)) * (sign(cos(t)))


def tangent(t):
    return sin(t) / cos(t)


def cotangent(t):
    return cos(t) / sin(t)


def weierstrass(t, n):
    return (a ** n) * cos((b ** n) * t)


def esin(t):
    return (euler ** (sin(t))) - (((1 / euler) + euler) / 2)


def sine(t):
    return sin(t)


def triangle(t):
    return (pi / 2) * arcsin(sin(t))


def sawtooth(t):
    return (((t / pi) - 1) % 2) - 1


def square(t):
    return sign(sin(t))


def pulse(t):
    return sign((cos(t) - cos(pi * pulsewidth)))


def semicircle(t):
    def sqrt(number):
        return number ** (1 / 2)

    radius = 1 / (4 * frequency)
    return (sign(sin(pi * t / (2 * radius))) *
            sqrt(radius ** 2 - (t % (2 * radius) - radius) ** 2)) / radius


def artifact(t):
    radius = frequency / 4
    return (sign(sin(((t * q) / omega) / 2 * radius)) * (
            (radius ** 2) - ((t * q) / omega % 2 * radius) ** (1 / 2)) / radius) - (
                   sign(sin(((t * q) / omega) / 2 * radius)) * (radius ** 2) - (
                   (((t * q) / omega) % (2 * radius)) ** (1 / 2)) / radius)


def nestedsine(t, n):
    return sin(sin(n * t))


def slx(t, n):
    return (sin(n * t)) / (n ** order)


def clx(t, n):
    return (cos(n * t)) / (n ** order)


def msaw(t, n):
    return (-(2 / pi)) * ((((-1) ** n) / n) * (sin(cos(n * (order / 100)) * (n * t))))


def skewer(t, n):
    return (-(2 / pi)) * ((((-1) ** n) / (2 * n + 1)) * (sin(cos(n * (order / 100)) * ((2 * n + 1) * t))))


def gaussiansine(t, n):
    return (euler ** (-n)) * sin(t * (euler ** n))


def logsin(t, n):
    return (log(n, order) * sin(log(n, order) * t)) / log(factorial(resolution), order)


def orderedsine(t):
    return (2 * (order ** sin(t)) / order) - 1


def antilogarithm(t):
    return (2 * (order ** (abs(sin(t)))) - order - 1) / (order - 1)


def tetration(t, n):
    return (n ** (-n)) * (sin(t * (n ** n)))


def errorsine(t, n):
    return (erf(n) * sin(t * erf(n))) / resolution


def decreasingfrequency(t, n):
    return (n * sin((t * resolution) / n)) / (resolution ** 2)


def randomsaw(t, n):
    return (-(2 / pi)) * ((((-1) ** n) / n) * (sin(((erf(.5+np.random.ranf(1)))+.1) * (n * t))))


print(semicircle(1 / q), "test")

SynthesisAlgorithm = {
    "atr": antitriangle,
    "cir": circular,
    "tan": tangent,
    "cot": cotangent,
    "wir": weierstrass,
    "esin": esin,
    "sin": sine,
    "tri": triangle,
    "saw": sawtooth,
    "sqr": square,
    "pls": pulse,
    "semi": semicircle,
    "nsin": nestedsine,
    "clx": clx,
    "slx": slx,
    "msaw": msaw,
    "skew": skewer,
    "gsin": gaussiansine,
    "lsin": logsin,
    "art": artifact,
    "osin": orderedsine,
    "alog": antilogarithm,
    "erfs": errorsine,
    "tetra": tetration,
    "decr": decreasingfrequency,
    "rsaw": randomsaw,
}
sinDenominator = {
    "cir": circular,
    "cot": cotangent
}
cosDenominator = {
    "tan": tangent
}
FourierFunctions = {
    "wir": weierstrass,
    "nsin": nestedsine,
    "gsin": gaussiansine,
    "erfs": errorsine,
}
SemiCircle = {
    "semi": semicircle,
    "art": artifact
}
OrderedFunctions = {
    "clx": clx,
    "slx": slx,
    "msaw": msaw,
    "skew": skewer,
    "lsin": logsin,
    "tetra": tetration,
    "decr": decreasingfrequency,
    "rsaw": randomsaw,
}
ModularFunctions = {
    "osin": orderedsine,
    "alog": antilogarithm
}
OneLowerBound = {
    "tetra": tetration,
    "decr": decreasingfrequency,
}

AlgorithmChosen = str(input("atr, clx, semi, cir, cot, esin, gsin, lsin, msaw,"
                            " nsin, pls, saw, sin, slx, skew, sqr, tan, tri, wir: "))
if AlgorithmChosen not in SynthesisAlgorithm:
    print("Error type, 'Undefined_Algorithm'")
    quit()

if AlgorithmChosen == "pls":
    pulsewidth = float(input("Pulsewidth: "))
if AlgorithmChosen in OrderedFunctions or ModularFunctions and not OneLowerBound:
    if AlgorithmChosen in ModularFunctions:
        if order < 0:
            print("Error type, 'Complex_Float'")
            quit()
        else:
            order = float(input("Enter order... "))

if AlgorithmChosen not in SemiCircle:
    for i in range(0, round(sampleTime * sampleLength)):
        if AlgorithmChosen in cosDenominator:
            if cos(i * omega) != 0:
                sample_data.append((SynthesisAlgorithm[AlgorithmChosen](i * omega)))
                percent(i)
            elif (i - (q / (4 * frequency))) % (q / 2 * frequency) == 0:
                sample_data.append(((SynthesisAlgorithm[AlgorithmChosen]((i - 1) * omega)) + (
                    SynthesisAlgorithm[AlgorithmChosen]((i + 1) * omega))) / 2)
                percent(i)
        elif AlgorithmChosen in sinDenominator:
            if sin(i * omega) != 0:
                sample_data.append((SynthesisAlgorithm[AlgorithmChosen](i * omega)))
                percent(i)
            elif i % (q / (2 * frequency)) == 0:
                sample_data.append(((SynthesisAlgorithm[AlgorithmChosen]((i - 1) * omega)) + (
                    SynthesisAlgorithm[AlgorithmChosen]((i + 1) * omega))) / 2)
                percent(i)
        elif AlgorithmChosen in FourierFunctions:
            sample_data.append((sum(SynthesisAlgorithm[AlgorithmChosen](i * omega, n) for n in range(resolution))))
            percent(i)
        elif AlgorithmChosen in OrderedFunctions:
            sample_data.append((sum(SynthesisAlgorithm[AlgorithmChosen](i * omega, k) for k in range(1, resolution))))
            percent(i)
        elif AlgorithmChosen in ModularFunctions:
            sample_data.append(SynthesisAlgorithm[AlgorithmChosen](i * omega))
            percent(i)
        else:
            sample_data.append((SynthesisAlgorithm[AlgorithmChosen](i * omega)))
            percent(i)
elif AlgorithmChosen in SemiCircle:
    for i in range(0, round(sampleTime * sampleLength)):
        sample_data.append(semicircle(i / q))
        percent(i)

# print(sample_data)

normalized_data = []
dataMax = max(sample_data)
dataMin = min(sample_data)
dataNormal = 0

if AlgorithmChosen not in SemiCircle:
    if dataMax >= abs(dataMin):
        dataNormal = dataMax
    else:
        dataNormal = abs(dataMin)

    if dataNormal >= 10:
        for z in sample_data:
            normalized_data.append(z)
    else:
        for z in sample_data:
            normalized_data.append(z / dataNormal)
else:
    if dataMax >= abs(dataMin):
        dataNormal = dataMax
    else:
        dataNormal = abs(dataMin)

    if dataNormal >= 10:
        for z in sample_data:
            normalized_data.append(z)
    else:
        for z in sample_data:
            normalized_data.append(z / dataNormal)

with open(r'FunctionGenerator.txt', 'w') as WS:
    print("Writing...")
    for WSD in normalized_data:
        WS.write("%s\n" % WSD)
    print(f"{str(AlgorithmChosen)} Sample Data Written!")

executionTime = round((time.time() - startTime), 2)
print('Execution time in seconds: ' + str(executionTime))
