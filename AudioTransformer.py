import wave
import numpy as np

with wave.open('ANALYTE.wav', 'rb') as inputWav:
    if inputWav.getnchannels() != 1:
        raise ValueError('Input WAV file must be mono')
    sampleRate = inputWav.getframerate()
    numFrames = inputWav.getnframes()
    inputBytes = inputWav.readframes(numFrames)
    analyte = np.frombuffer(inputBytes, dtype=np.int16)


def percent(progress):
    perc = 100 * (progress / float(len(analyte)))
    return print(f"\r{perc:.2f}%", end="")


def integration(analyte):
    n = len(analyte)
    cumulative_sum = np.empty(n, dtype=np.float64)
    cumulative_sum[0] = analyte[0]
    for i in range(1, n):
        cumulative_sum[i] = cumulative_sum[i - 1] + analyte[i]
        percent(i)
    return cumulative_sum


def remove_dc_offset(analyte_integral_modular):
    mean_value = np.mean(analyte_integral_modular)
    output_data = analyte_integral_modular - mean_value
    return output_data


analyte_integral = integration(analyte)
analyte_integral_modular = analyte_integral / np.max(np.abs(analyte_integral))
analyte_integral_modular_no_dc = remove_dc_offset(analyte_integral_modular)
analyte_derivative = np.diff(analyte)
analyte_derivative_normalized = analyte_derivative / np.max(np.abs(analyte_derivative))


def write_to_file(data, file_name, message):
    np.savetxt(file_name, data, fmt='%f')
    print(message)


write_to_file(analyte_derivative, 'ANALYTE_DERIVATIVE.txt', 'derivative Applied!')
write_to_file(analyte_integral_modular_no_dc, 'ANALYTE_INTEGRAL.txt', 'integral Applied!')


def write_wav(data, file_name, sample_rate, message):
    with wave.open(file_name, "w") as WAV:
        print(f"Writing {file_name}...")
        WAV.setnchannels(1)
        WAV.setsampwidth(2)
        WAV.setframerate(sample_rate)
        wav_data = (data * ((2 ** 15) - 1)).astype(np.int16)
        WAV.writeframes(wav_data.tobytes())
    print(message)


write_wav(analyte_derivative_normalized, "ANALYTE_DERIVATIVE.wav", sampleRate, "ANALYTE_DERIVATIVE.wav Written!")
write_wav(analyte_integral_modular_no_dc, "ANALYTE_INTEGRAL.wav", sampleRate, "ANALYTE_INTEGRAL.wav Written!")
