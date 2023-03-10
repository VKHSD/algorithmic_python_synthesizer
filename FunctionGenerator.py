import math
from math import log, erf
import numpy as np
import wave
import struct

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


def percent(progress):
    perc = 100 * (progress / float(sampleLength * sampleTime))
    return print(f"\r{perc:.2f}%", end="")


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


def bitcrush_algorithm(t):
    global bitcrush_signal
    if i % floor(q/(2*order)) == 0:
        bitcrush_signal = t
    return bitcrush_signal


def antitriangle(t):
    return ((sign(sin(arcsin(sin(t)))) * ((arcsin(cos(t))) ** 2)) / (pi * 2 * pi * frequency)) - (
            (sign(sin(arcsin(sin(t))))) / (8 * frequency))


def circular(t):
    return (((arcsin(sin(t))) / sin(t)) - (pi / 2)) * (sign(cos(t)))


def tangent(t):
    return sin(t / 2) / cos(t / 2)


def cotangent(t):
    return cos(t / 2) / sin(t / 2)


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
    return sin(sin(n * t)) / resolution


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
    return (2 * (order ** (abs(sin(t / 2)))) - order - 1) / (order - 1)


def antilogarithmsmooth(t):
    return (2 * (order ** (abs(sin((t % (q / (4 * frequency)) +
                                    (q / (8 * frequency))) / 2)))) - order - 1) / (order - 1)


def tetration(t, n):
    return (n ** (-n)) * (sin(t * (n ** n)))


def errorsine(t, n):
    return (erf(n) * sin(t * erf(n))) / resolution


def decreasingfrequency(t, n):
    return (n * sin((t * resolution) / n)) / (resolution ** 2)


def randomsaw(t, n):
    return (-(2 / pi)) * (
            (((-1) ** n) / n) * (sin((1 + ((np.random.ranf(1)) / (10 ** order))) * (n * t))))


def depthmod(t):
    return sin(t) - (((floor((2 ** (order - 1)) * sin(t))) + .5) / ((2 ** (order - 1)) - 1))


def phasemod(t):
    return (pi / 2) * (sin((order * t) + ((pi / 2) * sin((order + 1) * t))))


def anglemod(t):
    return sin(t) * cos(((order - 1) * t) + (order * sin(t)))


def randompulse(t):
    global random_pulse_width
    if (t / omega) <= 5:
        random_pulse_width = .5
    if (t / omega) % floor(q / (2 * frequency * floor(order))) == 0:
        if random_pulse_width <= .25:
            random_pulse_width = random_pulse_width + .1
        elif random_pulse_width >= .75:
            random_pulse_width = random_pulse_width - .1
        else:
            random_pulse_width = random_pulse_width + np.random.uniform(-.033, .033)
    return sign((sin(t) - sin(pi * random_pulse_width - .5)))


def fourierrandompulse(t, n):
    global fourier_random_pulse_width
    if (t / omega) <= 5:
        fourier_random_pulse_width = .5
    if (t / omega) % floor(q / (2 * frequency)) == 0:
        if fourier_random_pulse_width <= .25:
            fourier_random_pulse_width = fourier_random_pulse_width + .1
        elif fourier_random_pulse_width >= .75:
            fourier_random_pulse_width = fourier_random_pulse_width - .1
        else:
            fourier_random_pulse_width = fourier_random_pulse_width + np.random.uniform(-.033, .033)
    return fourier_random_pulse_width + (
            (2 / pi) * ((1 / n) * sin(pi * n * fourier_random_pulse_width) * cos(t * n))) - .5


def randomsquare(t):
    global rsqr_signal
    if (t / omega) <= 3:
        rsqr_signal = -1
    if (t / omega) % floor(q / (2 * frequency)) == 0:
        rsqr_signal = sign(np.random.uniform(-1, 1))
        if rsqr_signal == 0:
            rsqr_signal = -1
    return rsqr_signal


def bitcrush_sin(t):
    return bitcrush_algorithm(sin(t))


def bitcrush_sawtooth(t):
    return bitcrush_algorithm(sawtooth(t))


def bitcrush_triangle(t):
    return bitcrush_algorithm(triangle(t))


def bitcrush_antitriangle(t):
    return bitcrush_algorithm(antitriangle(t))


def bitcrush_circular(t):
    return bitcrush_algorithm(circular(t))


def hyperbolic_sin(t):
    return math.sinh(order*sin(t))/math.sinh(order)


def hyperbolic_tan(t):
    return math.tanh(order*sin(t))


def sineroot(t):
    if t/omega == 0:
        return 0
    else:
        sineroot_exponent = 1/(1+sin(order*t))
        abs_sine = abs(sin(t))
        return (abs_sine / sin(t))*(abs_sine ** sineroot_exponent)

# where all algorithms are stored
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
    "depth": depthmod,
    "phase": phasemod,
    "angle": anglemod,
    "rpulse": randompulse,
    "frpulse": fourierrandompulse,
    "rsqr": randomsquare,
    "bcsin": bitcrush_sin,
    "bctri": bitcrush_triangle,
    "bcsaw": bitcrush_sawtooth,
    "bcatr": bitcrush_antitriangle,
    "bccir": bitcrush_circular,
    "sinh": hyperbolic_sin,
    "tanh": hyperbolic_tan,
    "sinr": sineroot,
}
# where algorithms that divide by sine are stored unless parsed otherwise
sinDenominator = {
    "cir": circular,
    "cot": cotangent
}
# where algorithms that divide by cosine are stored unless parsed otherwise
cosDenominator = {
    "tan": tangent
}
# where algorithms that are a fourier series that begin at 0 are stored
FourierFunctions = {
    "wir": weierstrass,
    "nsin": nestedsine,
    "gsin": gaussiansine,
    "erfs": errorsine,
}
# where algorithms that are a fourier series that begin at 1 are stored
FourierOneFunctions = {
    "tetra": tetration,
    "decr": decreasingfrequency,
    "rsaw": randomsaw,
    "clx": clx,
    "slx": slx,
    "skew": skewer,
    "frpulse": fourierrandompulse,
}
# where algorithms that are a fourier series that begin at 1 and with an order modulator are stored
OrderedFourier = {
    "lsin": logsin,
    "msaw": msaw,
}
# a very special equation that needs a very special place
SemiCircle = {
    "semi": semicircle,
}
# where all ordered algorithms are stored so that the input for an order may be called
OrderedFunctions = {
    "clx": clx,
    "slx": slx,
    "msaw": msaw,
    "skew": skewer,
    "rsaw": randomsaw,
    "osin": orderedsine,
    "alog": antilogarithm,
    "alogsm": antilogarithmsmooth,
    "depth": depthmod,
    "phase": phasemod,
    "angle": anglemod,
    "rpulse": randompulse,
    "bcsin": bitcrush_sin,
    "bctri": bitcrush_triangle,
    "bcsaw": bitcrush_sawtooth,
    "bcatr": bitcrush_antitriangle,
    "bccir": bitcrush_circular,
    "sinh": hyperbolic_sin,
    "tanh": hyperbolic_tan,
    "sinr": sineroot,
}
# where algorithms that have an order modulator and that are NOT fourier functions
ModularFunctions = {
    "osin": orderedsine,
    "alog": antilogarithm,
    "depth": depthmod,
    "phase": phasemod,
    "angle": anglemod,
    "rpulse": randompulse,
    "bcsin": bitcrush_sin,
    "bctri": bitcrush_triangle,
    "bcsaw": bitcrush_sawtooth,
    "bcatr": bitcrush_antitriangle,
    "bccir": bitcrush_circular,
    "sinh": hyperbolic_sin,
    "tanh": hyperbolic_tan,
    "sinr": sineroot,
}
# another very special place for a very special algorithm
Alogsm = {
    "alogsm": antilogarithmsmooth,
}
# where all groups are placed
OtherGroups = (
    sinDenominator,
    cosDenominator,
    FourierOneFunctions,
    FourierFunctions,
    OrderedFourier,
    OrderedFunctions,
    ModularFunctions,
    SemiCircle,
    Alogsm,
)

AlgorithmChosen = str(input(f"\nalog, alogsm, angle, atr, bcatr, bccir, bcsaw, bcsin, bctri, cir, clx, decr, "
                            f"\ndepth, esin, erfs, frpulse, gsin, lsin, msaw, nsin, osin, pls, phase, rpulse, "
                            f"\nrsaw, rsqr, saw, semi, sin, sinh, sinr, skew, sqr, tetra, tan, tanh, wir"
                            f"\n"))

if AlgorithmChosen not in SynthesisAlgorithm:
    print("Error type, 'Undefined_Algorithm'")
    quit()

if AlgorithmChosen == "pls":
    pulsewidth = float(input("Pulsewidth: "))

if AlgorithmChosen in OrderedFunctions and AlgorithmChosen not in Alogsm:
    print("\nOrder may not be less than or equal to 1")
    mod_order = input("Enter Order... ")

    if mod_order.lower() == "u":
        order = 6.5737761766
    elif float(mod_order) > 1:
        order = float(mod_order)
    elif -1 < float(mod_order) <= 1:
        order = 1.0001
        print("Order = 1+...")
    elif float(mod_order) <= -1:
        order = frequency
        print(f"Order = {round(frequency, 4)}Hz")
    else:
        print("Error type, Invalid_Input")
        quit()
elif AlgorithmChosen in Alogsm:
    order = 6.5737761766
elif AlgorithmChosen in OrderedFourier:
    print("\nOrder may not be less than or equal to 1")
    mod_order = input("Enter Order... ")

    if mod_order.lower() == "u":
        order = 6.5737761766
    elif float(mod_order) > 1:
        order = float(mod_order)
    elif -1 < float(mod_order) <= 1:
        order = 1.0001
        print("Order = 1+...")
    elif float(mod_order) <= -1:
        order = frequency
        print(f"Order = {round(frequency, 4)}Hz")
    else:
        print("Error type, Invalid_Input")
        quit()

end = round(sampleTime * sampleLength)

for i in range(end):
    if AlgorithmChosen not in sinDenominator and AlgorithmChosen not in cosDenominator and AlgorithmChosen not in \
            FourierOneFunctions and AlgorithmChosen not in FourierFunctions and AlgorithmChosen not in OrderedFourier \
            and AlgorithmChosen not in OrderedFunctions and AlgorithmChosen not in ModularFunctions and \
            AlgorithmChosen not in SemiCircle and AlgorithmChosen not in Alogsm:
        sample_data.append(SynthesisAlgorithm[AlgorithmChosen](i * omega))
        percent(i)
    if AlgorithmChosen in cosDenominator:
        if cos(omega) != 0:
            sample_data.append(SynthesisAlgorithm[AlgorithmChosen](i * omega))
            percent(i)
        elif (i - (q / (4 * frequency))) % (q / 2 * frequency) == 0:
            sample_data.append((SynthesisAlgorithm[AlgorithmChosen](i * omega - 1) + SynthesisAlgorithm[
                AlgorithmChosen](i * omega + 1)) / 2)
            percent(i)
    elif AlgorithmChosen in sinDenominator:
        if sin(i * omega) != 0:
            sample_data.append(SynthesisAlgorithm[AlgorithmChosen](i * omega))
            percent(i)
        elif i % (q / (2 * frequency)) == 0:
            sample_data.append((SynthesisAlgorithm[AlgorithmChosen](i * omega - 1) + SynthesisAlgorithm[
                AlgorithmChosen](i * omega + 1)) / 2)
            percent(i)
    elif AlgorithmChosen in FourierFunctions:
        sample_data.append(sum(SynthesisAlgorithm[AlgorithmChosen](i * omega, n) for n in range(resolution)))
        percent(i)
    elif AlgorithmChosen in FourierOneFunctions:
        sample_data.append(sum(SynthesisAlgorithm[AlgorithmChosen](i * omega, n) for n in range(1, resolution)))
        percent(i)
    elif AlgorithmChosen in ModularFunctions:
        if AlgorithmChosen == antilogarithm:
            sample_data.append(SynthesisAlgorithm[AlgorithmChosen](i))
            percent(i)
        else:
            sample_data.append(SynthesisAlgorithm[AlgorithmChosen](i * omega))
            percent(i)
    elif AlgorithmChosen in Alogsm:
        sample_data.append(SynthesisAlgorithm[AlgorithmChosen](i * omega))
        percent(i)
    elif AlgorithmChosen in SemiCircle:
        sample_data.append(semicircle(i / q))
        percent(i)
    elif AlgorithmChosen in OrderedFourier:
        sample_data.append(sum(SynthesisAlgorithm[AlgorithmChosen](i * omega, n) for n in range(1, resolution)))
        percent(i)

normalized_data = []
dataMax = max(sample_data)
dataMin = min(sample_data)
dataNormal = max(dataMax, abs(dataMin))

if dataNormal >= 30:
    normalized_data = sample_data
else:
    normalized_data = [z / dataNormal for z in sample_data]

with open(r'FunctionGenerator.txt', 'w') as WS:
    print(f"\nWriting...")
    for WSD in normalized_data:
        WS.write("%s\n" % WSD)
    print(f"{str(AlgorithmChosen)} Sample Data Written!")

with wave.open(f"{AlgorithmChosen}.wav", "w") as WAVS:
    print(f"Writing {str(AlgorithmChosen)}.wav...")
    WAVS.setnchannels(1)
    WAVS.setsampwidth(2)
    WAVS.setframerate(q)
    for i in range(len(normalized_data)):
        percent(i)
        wav_data = int(normalized_data[i] * ((2 ** 15) - 1))
        WAVS.writeframes(struct.pack("<h", wav_data))

print(f"\n{str(AlgorithmChosen)}.wav Written!")
