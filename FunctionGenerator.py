import math
from math import log, erf
import numpy as np

a, b, q, euler, pi, order = 0.5, 5, 44100, 2.71828182, 3.14159265, 2.0
resolution = 100
midi_number = (input("Input MIDI #... C1 is 24... "))
exponent = ((float(midi_number) * 0.301029995664 - 57 * 0.301029995664) / 12)
frequency = 11 * 2 ** (exponent + 2) * 5 ** (exponent + 1)

print(round(frequency, 8), "Hz")
sampleLength = q
sampleTime = float(input("Length in seconds: "))
omega = (2 * pi * frequency) / q
sample_rate = round(q / frequency)
sample_data = []


def percent(iteration):
    if iteration % round(sampleLength / 10) == 0.0:
        return print(round((iteration / (sampleLength * sampleTime)) * 100, 2), "% done...") 


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
    return math.cos(x)


def sin(x):
    return math.sin(x)


def arcsin(x):
    return math.asin(x)


def arccos(x):
    return math.asin(x)


def arctan(x):
    return math.asin(x)


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
    return sin(t/2) / cos(t/2)


def cotangent(t):
    return cos(t/2) / sin(t/2)


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
    return (2 * (order ** (abs(sin(t/2)))) - order - 1) / (order - 1)


def antilogarithmsmooth(t):
    return (2 * (order ** (abs(sin((t % (q / (4 * frequency)) + (q / (8 * frequency))) / 2)))) - order - 1) / (order - 1)


def tetration(t, n):
    return (n ** (-n)) * (sin(t * (n ** n)))


def errorsine(t, n):
    return (erf(n) * sin(t * erf(n))) / resolution


def decreasingfrequency(t, n):
    return (n * sin((t * resolution) / n)) / (resolution ** 2)


def randomsaw(t, n):
    return (-(2 / pi)) * (
                (((-1) ** n) / n) * (sin((1 + ((np.random.ranf(1)) / (10 ** order))) * (n * t))))


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
    "osin": orderedsine,
    "alog": antilogarithm,
    "alogsm": antilogarithmsmooth,
    "tetra": tetration,
    "erfs": errorsine,
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
    "gsin": gaussiansine
}
SemiCircle = {
    "semi": semicircle
}
OrderedFunctions = {
    "clx": clx,
    "slx": slx,
    "msaw": msaw,
    "skew": skewer,
    "lsin": logsin,
    "tetra": tetration,
    "erfs": errorsine,
    "decr": decreasingfrequency,
    "rsaw": randomsaw,
}
ModularFunctions = {
    "osin": orderedsine,
    "alog": antilogarithm,
    "alogsm": antilogarithmsmooth,
    "rsaw": randomsaw,
}

AlgorithmChosen = str(input("alog, alogsm, atr, cir, clx, cot, decr, esin, gsin, lsin, msaw, nsin, osin, pls, rsaw, "
                            "saw, semi, sin, skew, slx, sqr, tan, tetra, tri."))
if AlgorithmChosen not in SynthesisAlgorithm:
    print("Error type, 'Undefined_Algorithm'")
    quit()

if AlgorithmChosen == "pls":
    pulsewidth = float(input("Pulsewidth: "))

if AlgorithmChosen in OrderedFunctions and AlgorithmChosen in ModularFunctions:
    if AlgorithmChosen in ModularFunctions:
        if AlgorithmChosen is not orderedsine:
            print("\nOrder may not be less than or equal to 1")
            mod_order = input("Constant of perfection of anti-logarithmic periodic waveforms is equal to "
                              "6.5737761766...\n"
                              "Please enter 'u' use this constant...")

            if mod_order.lower() == "u":
                order = 6.5737761766
            elif float(mod_order) > 1:
                order = float(mod_order)
            else:
                print("Error type, Invalid_Input")
                quit()
        else:
            order = float(input("Enter Order..."))
    else:
        order = float(input("Enter order... "))

    if order < 0:
        print("Error type, 'Complex_Float'")
        quit()


end = round(sampleTime * sampleLength)
if AlgorithmChosen not in SemiCircle:
    for i in range(end):
        if AlgorithmChosen in cosDenominator:
            if cos(omega) != 0:
                sample_data.append(SynthesisAlgorithm[AlgorithmChosen](i * omega))
                percent(i)
            elif (i - (q / (4 * frequency))) % (q / 2 * frequency) == 0:
                sample_data.append((SynthesisAlgorithm[AlgorithmChosen](i * omega - 1) + SynthesisAlgorithm[AlgorithmChosen](i * omega + 1)) / 2)
                percent(i)
        elif AlgorithmChosen in sinDenominator:
            if sin(i * omega) != 0:
                sample_data.append(SynthesisAlgorithm[AlgorithmChosen](i * omega))
                percent(i)
            elif i % (q / (2 * frequency)) == 0:
                sample_data.append((SynthesisAlgorithm[AlgorithmChosen](i * omega - 1) + SynthesisAlgorithm[AlgorithmChosen](i * omega + 1)) / 2)
                percent(i)
        elif AlgorithmChosen in FourierFunctions:
            sample_data.append(sum(SynthesisAlgorithm[AlgorithmChosen](i * omega, n) for n in range(resolution)))
            percent(i)
        elif AlgorithmChosen in OrderedFunctions:
            sample_data.append(sum(SynthesisAlgorithm[AlgorithmChosen](i * omega, k) for k in range(1, resolution)))
            percent(i)
        elif AlgorithmChosen in ModularFunctions:
            if AlgorithmChosen == antilogarithm:
                sample_data.append(SynthesisAlgorithm[AlgorithmChosen](i))
                percent(i)
            else:
                sample_data.append(SynthesisAlgorithm[AlgorithmChosen](i * omega))
                percent(i)
        else:
            sample_data.append(SynthesisAlgorithm[AlgorithmChosen](i * omega))
            percent(i)
elif AlgorithmChosen in SemiCircle:
    for i in range(end):
        sample_data.append(semicircle(i / q))
        percent(i)


normalized_data = []
dataMax = max(sample_data)
dataMin = min(sample_data)
dataNormal = max(dataMax, abs(dataMin))

if dataNormal >= 10:
    normalized_data = sample_data
else:
    normalized_data = [z / dataNormal for z in sample_data]

with open(r'FunctionGenerator.txt', 'w') as WS:
    print("Writing...")
    for WSD in normalized_data:
        WS.write("%s\n" % WSD)
    print(f"{str(AlgorithmChosen)} Sample Data Written!")

