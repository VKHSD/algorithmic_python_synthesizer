#                    )      )   (      (      
#                 ( /(   ( /(   )\ )   )\ )   
#       (   (     )\())  )\()) (()/(  (()/(   
#       )\  )\  |((_)\  ((_)\   /(_))  /(_))  
#      ((_)((_) |_ ((_)  _((_) (_))   (_))_   
#      \ \ / /  | |/ /  | || | / __|   |   \  
#       \ V /     ' <   | __ | \__ \   | |) | 
#        \_/     _|\_\  |_||_| |___/   |___/  



import os
import math
from math import log, erf
import numpy as np
import wave
import struct

# Folder name
folder_name = "GeneratedFunctions"

# Check if folder exists. If not, create it.
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

a, b, q, euler, pi, order = 0.5, 5, 44100, 2.71828182, 3.14159265, 2.0
resolution = 100
midi_number = 24  # (input("Input MIDI #... C1 is 24... "))
exponent = ((float(midi_number) * 0.301029995664 - 57 * 0.301029995664) / 12)
frequency = 11 * 2 ** (exponent + 2) * 5 ** (exponent + 1)

print(round(frequency, 8), "Hz")
sampleLength = q
sampleTime = 1  # float(input("Length in seconds: "))
omega = (2 * pi * frequency) / q
sample_rate = round(q / frequency)
sample_data = []


def handle_zero_division(func):
    """
    Decorator to handle ZeroDivisionError for the function it wraps.
    It will try to execute the function and if a ZeroDivisionError occurs,
    it will call the function again with the next value (t + 1).
    """

    def wrapper(t, *args, **kwargs):
        try:
            return func(t, *args, **kwargs)
        except ZeroDivisionError:
            return func(t + 1, *args, **kwargs)  # here, we're taking t+1 as the next best value

    return wrapper


@handle_zero_division
def percent(progress):
    perc = 100 * (progress / float(sampleLength * sampleTime))
    return print(f"\r{perc:.2f}%", end="")


@handle_zero_division
def floor(x):
    return int(x) if x >= 0 else int(x) - 1


@handle_zero_division
def ceil(x):
    return int(x) + 1 if x > int(x) else int(x)


@handle_zero_division
def sign(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1


@handle_zero_division
def cos(x):
    return math.cos(x)


@handle_zero_division
def sin(x):
    return math.sin(x)


@handle_zero_division
def arcsin(x):
    return math.asin(x)


@handle_zero_division
def arccos(x):
    return math.asin(x)


@handle_zero_division
def arctan(x):
    return math.asin(x)


@handle_zero_division
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


@handle_zero_division
def bitcrush_algorithm(t):
    global bitcrush_signal
    if i % floor(q / (2 * order)) == 0:
        bitcrush_signal = t
    return bitcrush_signal


@handle_zero_division
def antitriangle(t):
    return ((sign(sin(arcsin(sin(t)))) * ((arcsin(cos(t))) ** 2)) / (pi * 2 * pi * frequency)) - (
            (sign(sin(arcsin(sin(t))))) / (8 * frequency))


@handle_zero_division
def circular(t):
    return (((arcsin(sin(t))) / sin(t)) - (pi / 2)) * (sign(cos(t)))


@handle_zero_division
def tangent(t):
    return sin(t / 2) / cos(t / 2)


@handle_zero_division
def cotangent(t):
    return cos(t / 2) / sin(t / 2)


@handle_zero_division
def weierstrass(t, n):
    return (a ** n) * cos((b ** n) * t)


@handle_zero_division
def esin(t):
    return (euler ** (sin(t))) - (((1 / euler) + euler) / 2)


@handle_zero_division
def sine(t):
    return sin(t)


@handle_zero_division
def triangle(t):
    return (pi / 2) * arcsin(sin(t))


@handle_zero_division
def sawtooth(t):
    return (((t / pi) - 1) % 2) - 1


@handle_zero_division
def square(t):
    return sign(sin(t))


@handle_zero_division
def pulse(t):
    return sign((cos(t) - cos(pi * pulsewidth)))


@handle_zero_division
def semicircle(t):
    def sqrt(number):
        return number ** (1 / 2)

    radius = 1 / (4 * frequency)
    return (sign(sin(pi * t / (2 * radius))) *
            sqrt(radius ** 2 - (t % (2 * radius) - radius) ** 2)) / radius


@handle_zero_division
def nestedsine(t, n):
    return sin(sin(n * t)) / resolution


@handle_zero_division
def slx(t, n):
    return (sin(n * t)) / (n ** order)


@handle_zero_division
def clx(t, n):
    return (cos(n * t)) / (n ** order)


@handle_zero_division
def msaw(t, n):
    return (-(2 / pi)) * ((((-1) ** n) / n) * (sin(cos(n * (order / 100)) * (n * t))))


@handle_zero_division
def skewer(t, n):
    return (-(2 / pi)) * ((((-1) ** n) / (2 * n + 1)) * (sin(cos(n * (order / 100)) * ((2 * n + 1) * t))))


@handle_zero_division
def gaussiansine(t, n):
    return (euler ** (-n)) * sin(t * (euler ** n))


@handle_zero_division
def logsin(t, n):
    return (log(n, order) * sin(log(n, order) * t)) / log(factorial(resolution), order)


@handle_zero_division
def orderedsine(t):
    return (2 * (order ** sin(t)) / order) - 1


@handle_zero_division
def antilogarithm(t):
    return (2 * (order ** (abs(sin(t / 2)))) - order - 1) / (order - 1)


@handle_zero_division
def antilogarithmsmooth(t):
    return (2 * (order ** (abs(sin((t % (q / (4 * frequency)) +
                                    (q / (8 * frequency))) / 2)))) - order - 1) / (order - 1)


@handle_zero_division
def tetration(t, n):
    return (n ** (-n)) * (sin(t * (n ** n)))


@handle_zero_division
def errorsine(t, n):
    return (erf(n) * sin(t * erf(n))) / resolution


@handle_zero_division
def decreasingfrequency(t, n):
    return (n * sin((t * resolution) / n)) / (resolution ** 2)


@handle_zero_division
def randomsaw(t, n):
    return (-(2 / pi)) * (
            (((-1) ** n) / n) * (sin((1 + ((np.random.ranf(1)) / (10 ** order))) * (n * t))))


@handle_zero_division
def depthmod(t):
    return sin(t) - (((floor((2 ** (order - 1)) * sin(t))) + .5) / ((2 ** (order - 1)) - 1))


@handle_zero_division
def phasemod(t):
    return (pi / 2) * (sin((order * t) + ((pi / 2) * sin((order + 1) * t))))


@handle_zero_division
def anglemod(t):
    return sin(t) * cos(((order - 1) * t) + (order * sin(t)))


@handle_zero_division
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


@handle_zero_division
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


@handle_zero_division
def randomsquare(t):
    global rsqr_signal
    if (t / omega) <= 3:
        rsqr_signal = -1
    if (t / omega) % floor(q / (2 * frequency)) == 0:
        rsqr_signal = sign(np.random.uniform(-1, 1))
        if rsqr_signal == 0:
            rsqr_signal = -1
    return rsqr_signal


@handle_zero_division
def bitcrush_sin(t):
    return bitcrush_algorithm(sin(t))


@handle_zero_division
def bitcrush_sawtooth(t):
    return bitcrush_algorithm(sawtooth(t))


@handle_zero_division
def bitcrush_triangle(t):
    return bitcrush_algorithm(triangle(t))


@handle_zero_division
def bitcrush_antitriangle(t):
    return bitcrush_algorithm(antitriangle(t))


@handle_zero_division
def bitcrush_circular(t):
    return bitcrush_algorithm(circular(t))


@handle_zero_division
def hyperbolic_sin(t):
    return math.sinh(order * sin(t)) / math.sinh(order)


@handle_zero_division
def hyperbolic_tan(t):
    return math.tanh(order * sin(t))


@handle_zero_division
def sineroot(t):
    if t / omega == 0:
        return 0
    else:
        sineroot_exponent = 1 / (1 + sin(order * t))
        abs_sine = abs(sin(t))
        return (abs_sine / sin(t)) * (abs_sine ** sineroot_exponent)


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

MIN_ORDER = 1.0001

if AlgorithmChosen not in SynthesisAlgorithm:
    print("Error type, 'Undefined_Algorithm'")
    quit()

# Input for specific algorithm parameters
if AlgorithmChosen == "pls":
    pulsewidth = float(input("Pulsewidth: "))

if AlgorithmChosen in OrderedFunctions or AlgorithmChosen in OrderedFourier:
    print("\nOrder may not be less than or equal to 1")
    mod_order = input("Enter Order... ")

    if mod_order.lower() == "u":
        order = 6.5737761766
    else:
        mod_order = float(mod_order)
        if mod_order > 1:
            order = mod_order
        elif -1 < mod_order <= 1:
            order = MIN_ORDER
            print("Order adjusted to 1+...")
        else:
            order = frequency
            print(f"Order = {round(frequency, 4)}Hz")
elif AlgorithmChosen in Alogsm:
    order = 6.5737761766

end = round(sampleTime * sampleLength)
sample_data = []

for i in range(end):
    omega_i = i * omega

    # Handling functions with sine denominator issues
    if AlgorithmChosen in sinDenominator and sin(omega_i) == 0:
        sample_data.append(
            (SynthesisAlgorithm[AlgorithmChosen](omega_i - 1) + SynthesisAlgorithm[AlgorithmChosen](omega_i + 1)) / 2)
    # Handling functions with cosine denominator issues
    elif AlgorithmChosen in cosDenominator and cos(omega_i) == 0:
        sample_data.append(
            (SynthesisAlgorithm[AlgorithmChosen](omega_i - 1) + SynthesisAlgorithm[AlgorithmChosen](omega_i + 1)) / 2)
    # Handling Fourier functions
    elif AlgorithmChosen in FourierFunctions:
        sample_data.append(sum(SynthesisAlgorithm[AlgorithmChosen](omega_i, n) for n in range(resolution)))
    # Handling Fourier functions starting from 1
    elif AlgorithmChosen in FourierOneFunctions:
        sample_data.append(sum(SynthesisAlgorithm[AlgorithmChosen](omega_i, n) for n in range(1, resolution)))
    # Handling ordered Fourier functions
    elif AlgorithmChosen in OrderedFourier:
        sample_data.append(sum(SynthesisAlgorithm[AlgorithmChosen](omega_i, n) for n in range(1, resolution)))
    # Handling semi-circle functions
    elif AlgorithmChosen in SemiCircle:
        sample_data.append(semicircle(i / q))
    # Handling the Modular Functions and the special case for the antilogarithm
    elif AlgorithmChosen in ModularFunctions:
        if AlgorithmChosen == antilogarithm:
            sample_data.append(SynthesisAlgorithm[AlgorithmChosen](i))
        else:
            sample_data.append(SynthesisAlgorithm[AlgorithmChosen](omega_i))
    # For all other cases not specified above
    else:
        sample_data.append(SynthesisAlgorithm[AlgorithmChosen](omega_i))

    # Print progress in percentages at intervals (adjust the interval if needed)
    if i % 100 == 0:
        percent(i)

# Normalize
data_normal = max(max(sample_data), abs(min(sample_data)))
if data_normal < 30:
    sample_data = [z / data_normal for z in sample_data]

# Write to File
file_path = os.path.join(folder_name, 'FunctionGenerator.txt')
with open(file_path, 'w') as WS:
    WS.write('\n'.join(str(d) for d in sample_data))

wave_path = os.path.join(folder_name, f"{AlgorithmChosen}.wav")
with wave.open(wave_path, "w") as WAVS:
    WAVS.setnchannels(1)
    WAVS.setsampwidth(2)
    WAVS.setframerate(q)
    WAVS.writeframes(b''.join(struct.pack("<h", int(d * ((2 ** 15) - 1))) for d in sample_data))

print(f"\n{wave_path} Written!")
