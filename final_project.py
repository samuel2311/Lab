import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import soundfile as sf
import sounddevice as sd
from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy.io.wavfile import read
import os
from scipy import stats
from scipy.stats import skew
from scipy.stats import kurtosis
import pandas as pd

st.title("Proyecto Final - Señales y Sistemas")
st.write("Bienvenido a esta interfaz gráfica que le permitirá extaer carcterísticas en el dominio del tiempo y frecuencia de una señal mioeléctrica para reconocer el tipo de gesto contenido en la información.")


option = st.selectbox(
'Antes de comenzar, especifique el tipo de archivo que desea leer',
('-','CSV', 'WAV'))
uploaded_file = st.file_uploader("Seleccione un archivo de su computador para comenzar con el proceso:")
if uploaded_file is not None:

    st.write('Usted seleccionó:', option)
    if option == 'WAV':
        # To read file as bytes:
        y, fs = sf.read(uploaded_file, dtype='float32')
        t = len(y)/fs
        time_sf = np.arange(0,t, (1/fs))
    
    if option == 'CSV':
        y = np.loadtxt(uploaded_file, delimiter=",",skiprows=1)
        fs = 200
        t = len(y)/fs
        time_sf = np.arange(0,t, (1/fs))

    #FFT
    fft_0 = np.fft.fft(y)
    fft = abs(np.fft.fft(y))
    f = np.fft.fftfreq(fft.size)*fs

    #ploting
    fig, (ax1,ax2) = plt.subplots(2, figsize=(10, 8))
    ax1.plot(time_sf,y)
    ax1.set_title("Señal en el dominio del tiempo")
    ax1.set(xlabel='t (s)', ylabel='x(t)')
    ax2.stem(f, fft)
    ax2.set_title("Señal en el dominio de la frecuencia")
    ax2.set(xlabel='f (Hz)', ylabel='FFT')
    fig.tight_layout()

    st.subheader("Diagrama de la señal en el dominio del tiempo y la frecuencia")
    st_a = st.pyplot(plt)

    N = len(fft)
    n = len(y)

    for i in range(len(y)):
        for j in range(i+1,len(y)):
            if y[i]>y[j]:
                y[i],y[j]=y[j],y[i]

    for i in range(len(fft)):
        for j in range(i+1,len(fft)):
            if fft[i]>fft[j]:
                fft[i],fft[j]=fft[j],fft[i]                   

    #media FFT
    u = 0
    for j in range(N):
        u = u + (fft[j]/N)
    # st.write(u)

    # u2 = np.mean(fft)
    # st.write(u2)
    
    #media time
    u1 = 0
    for j in range(n):
        u1 = u1 + (y[j]/n)
    # st.write(u1)

    # u3 = np.mean(y)
    # st.write(u3)


    
    #mediana
    if N % 2 == 0:
        a = int(N/2)
        b = int((N/2)+1)
        
        A = fft[a]
        B = fft[b]
        m = (A+B)/2
        # st.write(m)
        
        # m1 = np.median(fft)
        # st.write(m1)
    else:
        m = (fft[N]+1)/2
        st.write(m)
        m1 = np.median(fft)
        st.write(m1)

    if n % 2 == 0:
        a1 = int(n/2)
        b1 = int((n/2)+1)
        
        A1 = y[a1]
        B1 = y[b1]
        m1 = (A1+B1)/2
        # st.write(m1)

        # m2 = np.median(y)
        # st.write(m2)
    else:
        m1 = (fft[n]+1)/2
        st.write(m)
        m2 = np.median(y)
        st.write(m2)

    #Standar deviation FFT
    de = 0
    for j in range(n):
        de = de + (((abs(fft[j])-u)**2)/N)
    de = de**0.5
    # st.write(de)

    # de2 = np.std(fft)
    # st.write(de2)
    
    #Standar deviation time
    de1 = 0
    for j in range(n):
        de1 = de1 + (((y[j]-u1)**2)/n)
    de1 = de1**0.5
    # st.write(de1)

    # de3 = np.std(y)
    # # st.write(de3) 

    #Desviación media absoluta
    dm = 0
    for j in range(N):
        dm = dm + abs((fft[j]-u)/N)
    # st.write(dm)

    # dm2 = np.mean(abs(fft - u))
    # st.write(dm2)
    
    #Desviación media absoluta time
    dm1 = 0
    for j in range(n):
        dm1 = dm1 + abs((y[j]-u1)/n)
    # st.write(dm1)

    # dm3 = np.mean(abs(y - u1))
    # st.write(dm3)

    #Cuartil 25 FFT
    c25 = int(N/4)
    c25f = fft[c25]
    # st.write(c25f)

    #Quartil 25 time
    c25n = int(n/4)
    c25t = y[c25n]
    # st.write(c25t)

    #Cuartil 75 FFT
    c75 =int((3*N)/4)
    c75f = fft[c75]
    # st.write(c75f)

    #cuartil 75 time
    c75n = int((3*n)/4)
    c75t = y[c75n]
    # st.write(c75t)

    #IQR
    IQRf = c75f -c25f
    # st.write(IQRf)

    IQRt = c75t -c25t
    # st.write(IQRt)

    # IQRt = stats.iqr(fft, interpolation = 'midpoint')
    # st.write(IQRt)
    
    # IQR = stats.iqr(y, interpolation = 'midpoint')  
    # st.write(IQR)

    #Asimetría
    As = 0
    for j in range(N):
        As = As + (((fft[j]-u)**3)/N)
    As = As/(de**3)
    # st.write(As)

    # as3 = skew(abs(fft), axis=0, bias=True)
    # st.write(as3)

    #Asimetría time
    As2 = 0
    for j in range(n):
        As2 = As2 + (((y[j]-u1)**3)/n)
    As2 = As2/(de1**3)
    # st.write(As2)

    # as4 = skew(y, axis=0, bias=True)
    # st.write(as4)

    #Curtosis FFT
    Cu = 0
    # for j in range(N):
    #     Cu = Cu + (((fft[j]-u)**4)/N)
    # Cu = Cu/(de**4)
    # st.write(Cu)

    Cu = kurtosis(fft, axis=0, bias=True)
    # st.write(Cu)

    #Curtosis time
    Cu2 = 0
    # for j in range(n):
    #     Cu2 = Cu2 + (((y[j]-u1)**4)/n)
    # Cu2 = Cu2/(de1**4)
    # st.write(Cu2)

    Cu1 = kurtosis(y, axis=0, bias=True)
    # st.write(Cu1)

    #Coeficiente de variación FFT
    cv = de/u
    # st.write(cv)

    #coeficiente de variación time 
    cvt = de1/u1
    # st.write(cvt) 

    #Potencia FFT
    P0 = 0
    for j in range(-N,N):
        P0 = P0 + (abs(fft[j]**2))
    P = (1/(2*N+1))*P0
    # st.write(P)

    #Potencia time
    P01 = 0
    for j in range(-n,n):
        P01 = P01 + (abs(y[j]**2))
    P1 = (1/(2*n+1))*P01
    # st.write(P1)

    #Energía de Shannon
    E = 0
    for i in range(N):
        E = E + (fft[i]**2)*np.log((fft[i]+0.000001)**2)
    # st.write(E)  

    #Energía Shannon time
    E1 = 0
    for i in range(0,n,1):
        E1 = E1 + (y[i]**2)*np.log((y[i]+0.000001)**2)
    # st.write(E1)     

    par = ['Media', 
    'Mediana', 
    'Desviacón estándar', 
    'Desviación media absoluta',
    'Cuartil 25 (Q1)',
    'Cuartil 75 (Q3)',
    'IQR',
    'Asimetría',
    'Curtosis',
    'Coeficiente de variación',
    'Potencia',
    'Energía de Shannon']

    para = [u1,m1,de1,dm1,c25t,c75t,IQRt,As2,Cu1,cvt,P1,E1]
    para2 = [u,m,de,dm,c25f,c75f,IQRf,As,Cu,cv,P,E] 
    
    df = pd.DataFrame(list(zip(par, para, para2)), columns=['Parámetros','Dominio del Tiempo','Dominio de la Frecuencia'])

    st.subheader("Características extraídas de la señal")
    st.table(df)

    st.title("El gesto correspondiente a la señal ingresada es:")

    if (-0.802734375 <= u1 <= 0.14260) and (-2.0 <= m1 <= 1) and (7.03800 <= de1 <= 38.89870) and (5.57710 <= dm1 <= 30.315940856933594) and (-25 <= c25t <= -5) and (4 <= c75t <= 23) and (9 <= IQRt <= 48) and (-0.60140 <= As2 <= 0.533706371498648) and (0.09907778442955051 <= Cu1 <=  1.9896929244536630) and (-48.45772725887586 <= cvt <= 132.47881606061654) and (50.048780487804876 <= P1 <= 1512.2751219512195) and (119626.94 <= E1 <= 6285003.11):
        st.info("Paper")
    elif (-5.669921875 <= u1 <= 0.2227) and (-6 <= m1 <= 0 )and (3.7516674149415197 <= de1 <= 60.0104) and (2.8044357299804688 <= dm1 <= 47.4030) and (-45 <= c25t <= -3) and (1 <= c75t <= 34) and (4 <= IQRt <= 79) and (-1.314 <= As2 <= 0.8435492156444461) and (-0.36706281349821257 <= Cu1 <=  8.8721) and (-38.3701 <= cvt <= 146.15590082687064) and (14.6439 <= P1 <= 3629.847804878049) and (2551122.54 <= E1 <= 16575374.833):
        st.info("Rock")
    else:
        st.write("Ha ingresado la señal de un gesto diferente")
    with st.expander("Ver explicación"):

        st.write("Para determinar si una señal corresponde al gesto 'paper' o 'rock' se tienen en cuenta características propias de cada gesto previamente registradas en una base de datos.")

        par = ['Media', 
        'Mediana', 
        'Desviacón estándar', 
        'Desviación media absoluta',
        'Cuartil 25 (Q1)',
        'Cuartil 75 (Q3)',
        'IQR',
        'Asimetría',
        'Curtosis',
        'Coeficiente de variación',
        'Potencia',
        'Energía de Shannon']

        para = ["-0,80270 <= x <= 0,14260", "-2 <= x <= 1", "7,03800 <= x <= 38,89870", "5,57710 <= x <= 30,31590", "-25 <= x <= -5", "4 <= x <= 23", "9 <= x <= 48", "-0,60140 <= x <= 0,53370", "0,09910 <= x <=  1,98970", "-48,45770 <= x <= 132,47880", "50,04880 <= x <= 1512,27510","119626,95 <= x <= 6285003,10790"]
        para2 = ["136,821 <= x <= 743,7421", "118,0602 <= x <= 645,2101", "83,2471 <= x <= 471,0507","65,089 <= x <= 378,927", "75,135 <= x <= 389,338", "183,873 <= x <= 1043,555", "107,4021 <= x <= 654,216", "0,7767 <= x <= 1,2652", "2,9567 <= x <= 4,9606", "0,6015 <= x <= 0,65420", "25624,9756 <= x <= 774284,8624", "186909182.34190 <= x <= 672846682,8117"] 

        df = pd.DataFrame(list(zip(par, para, para2)), columns=['Parámetros','Dominio del Tiempo','Dominio de la Frecuencia'])

        st.subheader("Rango de características del gesto 'Paper'")
        st.table(df)

        par = ['Media', 
        'Mediana', 
        'Desviacón estándar', 
        'Desviación media absoluta',
        'Cuartil 25(Q1)',
        'Cuartil 75 (Q3)',
        'IQR',
        'Asimetría',
        'Curtosis',
        'Coeficiente de variación',
        'Potencia',
        'Energía de Shannon']

        para = ["-5,6699 <= x <= 0,2227", "-6 <= x <= 0", "3,7517 <= x <= 60,0104", "2,8044 <= x <= 47,4030", "-45 <= x <= -3", "1 <= x <= 34", "4 <= x <= 79", "-1,314 <= x <= 0,8435", "-0,367 <= x <=  5,8721", "-38,3701 <= x <= 146,1559", "14,6439 <= x <= 3629,8478","2551122,545 <= x <= 16575374,8328"]
        para2 = ["73,93100 <= x <= 1159,9549", "67,67320 <= x <= 1017,4467", "45,1576 <= x <= 717,497", "273,4995 <= x <= 277,1582", "196,8502 <= x <= 266,404", "690,8627 <= x <= 773,3018", "494,0124 <= x <= 506,8978", "1,0231 <= x <= 0,7267", "4,1616 <= x <= 3,0158", "0,7174 <= x <= 0,6302", "342156,8624 <= x <= 396908,3941",11] 

        df = pd.DataFrame(list(zip(par, para, para2)), columns=['Parámetros','Dominio del Tiempo','Dominio de la Frecuencia'])

        st.subheader("Rango de características del gesto 'Rock'")
        st.table(df)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col5:
            st.markdown('[Base de datos](https://uninorte-my.sharepoint.com/:x:/g/personal/barriossd_uninorte_edu_co/EY8kcmLOsYhChlY6qqX25xoBotZyrRuq2U7ci2XswDRpJQ?e=vWt7bt)')

    # Implementation of matplotlib function
    fig2, (spec) = plt.subplots(figsize=(10, 6))
    spec.specgram(fft_0, NFFT= 128, Fs=fs, noverlap= 120, cmap='jet_r')
    spec.set_title("Espectograma")
    spec.set(xlabel='t(s)', ylabel='freq (Hz)')

    st.subheader("Espectograma de la señal")
    st.pyplot(fig2)

