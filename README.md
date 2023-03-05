#AudioTransformer
This takes a MONO input .wav file and outputs a normalized version of the input, differentiated input, and integrated input. This gives the integral of the function which leaves a bass-heavy signal and a differentiated signal which also provides a treble-heavy version. It turns triangle waves into square waves when differentiated and turns square waves into triangle waves. 

The input file must be named "ANALYTE" and a .wav file of the type 16bit. 

##########################################################################################################################################################

# algorithmic_python_synthesizer
Many algorithms to choose from, select a midi number, length, and synthesis algorithm of your choice. 

Full list of working algorithms:

    "atr": antitriangle,
    
    # the antiderivative of the triangle wave. has good bass tonality.
    
    "cir": circular,
    
    # a sort of circular waveform I discovered.
    
    "tan": tangent,
    
    # sin / cos.
    
    "cot": cotangent,
    
    # cos / sin.
    
    "wir": weierstrass,
    
    # continuous but non-differentiable wave. has even spreading of overtones.
    
    "esin": esin,
    
    # eulers constant to the power of sin.
    
    "sin": sine,
    
    # a simple sine function used for calibration.
    
    "tri": triangle,
    
    # a triangle wave made from trigonometric functions, i.e., arctan(sin).
    
    "saw": sawtooth,
    
    # sawtooth made from the mod operator.
    
    "sqr": square,
    
    # a square wave made from the sign operator. i.e., sign(sin).
    
    "pls": pulse,
    
    # a pulse wave made from the sign operator. (similar to the square.
    
    "semi": semicircle,
    
    # a semi circle operated with mod and sign to keep a continuous waveform.
    
    "nsin": nestedsine,
    
    # sin(sin).
    
    "clx": clx,
    
    # Clausen's function of cosine.
    
    "slx": slx,
    
    # Clausen's function of sine.
    
    "msaw": msaw,
    
    # modulated saw, creates noise the longer the waveform.
    
    "skew": skewer,
    
    # a skewed square wave with an order variable. creates noise the longer the waveform.
    
    "gsin": gaussiansine,
    
    # a sine function operated by gaussian distribution for frequency distribution.
    
    "lsin": logsin,
    
    # logarithmic distribution of frequency.
    
    "art": artifact,
    
    # currently unused but remains as an artifact of what once was a very broken semi-circle.
    
    "osin": orderedsine,
    
    # the ordered sin is an order to the power of sin. similar to the esin but order is a variable that is interchangable.
    
    "alog": antilogarithm,
    
    # the antilogarithm is the inverse of logarithmic distortion as a function.
    
    "alogsm": antilogarithmsmooth
    
    # antilogarithmsmooth is the smoothest antilogarithm possible, by use of constants derived by me, we were able to achieve a continuous waveform. something broke during optimization though, and i am working on a fix. single cycle waveforms are doable easily, but the longer the waveform goes on, the more clicks occur. 
    
    "erfs": errorsine,
    
    # the use of the error function to distribute overtones.
    
    "tetra": tetration,
    
    # the use of a tetration to distribute overtones.
    
    "decr": decreasingfrequency,
    
    # best use case if starting at a high frequency, the decreasingfrequency algorithm generates a waveform that distributes frequency that is lowering in pitch. it is more of a noise generator than a tonal device.
    
    "rsaw": randomsaw,
    
    # the random saw uses slight variation in pitch that creates noise the longer a waveform goes.
    
    "depth": depthmod,
    
    # uses a hypothetical pure sinusoid and takes the difference of a digital sinusoid of bit depth of (order - 1).
    
    "phase": phasemod,
    
    # uses basic phase modulation to generate a wave. as in. using an oscillator for a phase change
    
    "angle": anglemod,
    
    # uses angle modulation. as in, modulating multiple angles using an order and other oscillators.
    
    "rpulse": randompulse,
    
    # sends a new pulsewidth variable for every phase cycle multiplied by order.

    "frpulse": fourierrandompulse,
    
    # same idea as the rpulse algorithm except using a fourier series. (currently not working as intended).
    
    "rsqr": randomsquare,
    
    # sends either a HIGH or LOW signal based on a 50% chance every theoretical cycle.
    
    "bcsin": bitcrush_sin,
    
    # bitcrush of the sin wave as order = frequency in which the bitcrush with crush.
    
    "bctri": bitcrush_triangle,
    
    # bitcrush of the triangle wave as order = frequency in which the bitcrush with crush.
    
    "bcsaw": bitcrush_sawtooth,
    
    # bitcrush of the sawtooth wave as order = frequency in which the bitcrush with crush.
    
    "bcatr": bitcrush_antitriangle,
    
    # bitcrush of the antitriangle wave as order = frequency in which the bitcrush with crush.
    
    "bccir": bitcrush_circular,
    
    # bitcrush of the circular wave as order = frequency in which the bitcrush with crush.
    
    "sinh": hyperbolic_sin,
    
    # sinh(order*sin) makes for an interesting tone
    
    "tanh": hyperbolic_tan,
    
    # tanh(order*sin) gives a nice distorted tone
    
    "sinr": sineroot,
    
    # a big mess of sines that gives a very odd modulation of tone.
