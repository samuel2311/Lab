
from matplotlib import pyplot as plt
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, freqz
from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy.io.wavfile import read
from scipy.signal import upfirdn
from pydub import AudioSegment
from pydub.utils import make_chunks

st.title("Procesamiento de señales de audio")
st.subheader("Suma de dos señales de audio y filtrado de ruido")

st.sidebar.title("Bono - primer parcial")

st.sidebar.markdown("Se tienen dos señales de audio; la voz y la pista de una canción. Pero, una de las dos señales fue grabada accidentalmente con el doble de frecuencia de la otra.")

#show both audios
audio_file = open('C:/Users/Samue/Music/pista.wav', 'rb')
audio_bytes = audio_file.read()

st.sidebar.audio(audio_bytes, format='audio/wav')

audio_file2 = open('C:/Users/Samue/Music/voz.wav', 'rb')
audio_bytes2 = audio_file2.read()

st.sidebar.audio(audio_bytes2, format='audio/wav')

st.sidebar.markdown("Se necesita pasar la canción por un sistema de filtrado de ruido, para esto, se debe sumar tanto la pista de la canción como la voz.")

#read
framerate_sf, signal = wavfile.read('C:/Users/Samue/Music/pista.wav')
t = len(signal)/framerate_sf
time_sf = np.arange(0,t, (1/framerate_sf))

filas = st.columns(2)

with filas[0]:
    st.markdown("La frecuencia de muestreo de la pista de audio es: ")
    st.text(framerate_sf)
    st.markdown("Formato: ")
    st.text(signal.dtype)

framerate_sf2, signal2 = wavfile.read('C:/Users/Samue/Music/voz.wav')
t2 = len(signal2)/framerate_sf2
time_sf2 = np.arange(0,t2,(1/framerate_sf2))

with filas[1]:
    st.markdown("La frecuencia de muestreo de la pista de voz es: ")
    st.text(framerate_sf2)
    st.markdown("Formato: ")
    st.text(signal2.dtype)

fig,(graph1, graph2) = plt.subplots(2,1, figsize=(15, 6))


plt.ylabel('Amplitud')
plt.xlabel('Tiempo (segundos)')
plt.title('Amplitud vs Tiempo')
plt.ylabel('Amplitud')
plt.xlabel('Tiempo (segundos)')
graph1.plot(time_sf, signal, alpha=0.5)
graph2.plot(time_sf2, signal2, alpha=0.5)
graph1.title.set_text('Pista de audio')
graph2.title.set_text('Grabación de voz')
st_a= st.pyplot(plt)

st.markdown("Pulse el botón para ejecutar el programa de correción y procesamiento de señales")
pos_boton=st.columns(7)

with pos_boton[3]:
    boton = st.button('Ejecutar')
if boton:
    #upsampling
    signal3 = upfirdn([1], signal2, 2)
    framerate_sf3 = framerate_sf2*2
    
    t3 = len(signal3)/framerate_sf3
    time_sf3 = np.arange(0,t3, (1/framerate_sf3))
    
    len = signal3.shape[0]/framerate_sf3
    filas = st.columns(2)
    with filas[1]:
        st.markdown("Al corregir la frecuencia de muestreo, la nueva duración de la grabación de voz es: ")
        st.text(signal3.shape[0]/framerate_sf3)
    with filas[0]:
        st.markdown("La frecuencia de muestreo de la grabación de voz ahora es: ")
        st.text(framerate_sf3)

    st.subheader("Grabación de voz con frecuencia corregida")
    write('vozcorregida.wav', framerate_sf3, signal3.astype(np.int16))
    audio_file11 = open('vozcorregida.wav', 'rb')
    audio_bytes11 = audio_file11.read()

    st.audio(audio_bytes11, format='audio/wav')


    fig,(graph0,graph) = plt.subplots(2,1, figsize=(15, 6))

    plt.ylabel('Amplitud')
    plt.xlabel('Tiempo (segundos)')
    graph.plot(time_sf3, signal3, alpha=0.5)
    graph0.plot(time_sf, signal, alpha=0.5)
    graph.title.set_text('Señal con ruido')
    graph0.title.set_text('Señal con ruido')

    st_a1= st.pyplot(plt)

    #suma de señales
    st.markdown("Una vez las dos señales están muestreadas a la misma frecuencia, se pueden sumar.")

    sum=signal[:264599]+signal3[:264599]

    fig, (graph0) = plt.subplots(1,1,figsize=(15, 3))
    plt.title('Señal de audio y voz')
    plt.ylabel('Amplitud')
    plt.xlabel('Tiempo [segundos]')
    graph0.plot(time_sf3, sum, alpha=0.5)
    st_a2= st.pyplot(plt)
    
    write('pistayvoz.wav', framerate_sf3, sum.astype(np.int16))

    st.subheader("Suma de la pista de audio y la grabación de voz")
    audio_file22 = open('pistayvoz.wav', 'rb')
    audio_bytes22 = audio_file22.read()

    st.audio(audio_bytes22, format = 'audio/wav')

    #Filtrado (filtro pasa baja)
    st.markdown("La suma de las dos señales se pasa por un filtro pasa baja que elimina el ruido de la señal.")
    def butter_lowpass(cutoff, fs, order=5):
        nyqu = 0.5 * fs
        normal_cutoff = cutoff / nyqu
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y


    # Setting standard filter requirements.
    order = 6
    fs = 30.0       
    cutoff = 3.667  

    b, a = butter_lowpass(cutoff, fs, order)

    # Plotting the frequency response.
    w, h = freqz(b, a, worN=8000)

    # Filtering and plotting
    y = butter_lowpass_filter(sum, cutoff, fs, order)

    fig, (graph222, graph333) = plt.subplots(2, 1, figsize=(15, 6))
    st_b = st.pyplot(plt)

    graph222.plot(time_sf[:264599], sum[:264599], label='data')
    graph333.plot(time_sf[:264599], y[:264599], label='filtered data')
    graph222.title.set_text('Señal con ruido')
    graph333.title.set_text('Señal filtrada')
    plt.ylabel('Amplitud')
    plt.xlabel('Tiempo [segundos]')

    st_b.pyplot(plt)

    write('nuevapista.wav', framerate_sf3, y.astype(np.int16))

    audio_file3 = open('nuevapista.wav', 'rb')
    audio_bytes3 = audio_file3.read()
    

    st.subheader("La señal procesada y filtrada es la siguiente: ")
    st.audio(audio_bytes3, format='audio/wav')
    st.caption("Lorde - Stoned at the nail salon (2021)")