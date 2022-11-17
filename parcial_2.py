import streamlit as st
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy.io.wavfile import read

st.title("Parcial 2 - Señales y Sistemas")
st.subheader("Punto 1")
st.markdown("¿Recuerda la canción del parcial anterior?... Dicha canción presentaba un ruido que necesitaba ser filtrado. Adjunto encontrará un archivo de audio (song.wav) que contiene un fragmento de la pista de la canción donde se puede escuchar el sonido de un bajo y un ruido no deseado con un tono más agudo, lea el archivo utilizando Python o Matlab y elimine el ruido que presenta la grabación utilizando lo aprendido en la asignatura")
audio_file = open('C:/Users/Samue/Downloads/song.wav', 'rb')
audio_bytes = audio_file.read()

st.subheader("Señal original con ruido: ")
st.audio(audio_bytes, format='audio/wav')

framerate_sf, signal = wavfile.read('C:/Users/Samue/Downloads/song.wav')
t = len(signal)/framerate_sf
time_sf = np.arange(0,t, (1/framerate_sf))
st.write(signal.dtype)
#FFT
fft = np.fft.fft(signal)
f1 = np.fft.fftfreq(fft.size)*framerate_sf

fig, (ax1, ax2) = plt.subplots(2)
st_a = st.pyplot(plt)
ax1.plot(time_sf, signal)
ax1.set_title("Señal original en el dominio del tiempo")
ax1.set(xlabel='t', ylabel='x(t)')
ax2.plot(f1, abs(fft))
ax2.set_title("Señal original en el dominio de la frecuencia")
ax2.grid()
ax2.set(xlabel='f', ylabel='FFT')
fig.tight_layout()
st_a.pyplot(plt)



#filtro
st.subheader("Filtro")
st.markdown("Se genera un escalón unitario que, al operar con la FFT de la señal original, permita elimianr las partes indeseadas.")
u2 = lambda t: np.piecewise(t,t>=2,[1,0])
u4 = lambda t: np.piecewise(t,t>=4,[1,0])
rectangular = lambda t:u2(t) - u4(t)
rect_i = rectangular((f1/500)+2)

fig6, graph10 =plt.subplots()
st_h = st.pyplot(plt)
plt.plot(f1,rect_i)
graph10.set(xlabel='f', ylabel='FFT')
plt.grid()

st_h.pyplot(plt)

#Convolución en frecuencia
st.subheader("Operación de las dos señales en frecuencia")
st.markdown("Al operar nuestro filtro con la señal original se obtiene la señal resultante.")
conv = fft*rect_i
st.write(conv.dtype)
fig3, ax3 = plt.subplots()
st_c= st.pyplot(plt)
ax3.plot(f1, abs(conv))
ax3.set(xlabel='f', ylabel='FFT')
ax3.grid()

st_c.pyplot(plt)


n_audio = np.fft.ifft(conv)

st.subheader("Devolvemos nuestra señal al dominio del tiempo")
fig4, ax5 = plt.subplots(1,1,figsize=(10, 5))
st_e= st.pyplot(plt)
ax5.plot(time_sf, n_audio)
ax5.set(xlabel='t', ylabel='x(t)')

st_e.pyplot(plt)

write('nuevapista.wav', framerate_sf, n_audio.astype(np.int16))

audio_file3 = open('nuevapista.wav', 'rb')
audio_bytes3 = audio_file3.read()


st.subheader("La señal procesada y filtrada es la siguiente: ")
st.audio(audio_bytes3, format='audio/wav')


