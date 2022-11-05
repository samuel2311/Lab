import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image

image = Image.open('C:/Users/Samue/Downloads/Logo-UNINORTE.png')

st.sidebar.image(image)
st.sidebar.title("Laboratorio 3. *Señales y sistemas*")

option = ['-', 'Series de Fourier', 'Transformada de Fourier']
opt0 = st.sidebar.selectbox("Seleccione la opción con que desea trabajar", option)

if opt0 == 'Series de Fourier':
    st.title("Series de Fourier")
    st.latex("x(t) = A_{0} + \sum_{n=1}^{n=∞} A_{n}*cos(W_{o}kt) + B_{n}*sen(W_{o}kt)")
    st.caption("Cualquier forma de onda con energía finita en un intervalo dado, se puede expresar mediante una suma infinita de exponenciales complejas que oscilan a diferentes frecuencias. [1, secc. 3]")
    señales = ['-', 'Rectangular periódica', 'Triangular', 'Exponencial', 'Sinusoidal rectificada', 'Rampa trapezoidal']
    opt = st.sidebar.selectbox('Seleccione una opción para representar en series de Fourier', señales)

    fig, (graph) = plt.subplots(nrows=1,ncols=1)
    st_a = st.pyplot(plt)

    st.write("Presione el siguiente botón para ejecutar el programa de reconstrucción de señales con Series de Fourier.")
    pos_boton=st.columns(7)
    with pos_boton[3]:
        boton = st.button('Ejecutar')

    if opt == 'Triangular':
        a = st.sidebar.number_input("Digite la amplitud de la señal: ")
        n = st.sidebar.number_input("Digite el número de armónicos: ")
        p = st.sidebar.number_input("Ingrese el periodo: ")
        if p>0:
            n = n+1
            wo = (2*np.pi)/p
            dt = 0.0001
            t = np.arange(0,int(p),dt)
            y = a*signal.sawtooth(wo*t, 0.5)
            
            graph.plot(t,y)
            plt.xlabel("t(s)")
            plt.ylabel("x(t)")
            plt.title("Gráfica x(t)")
            max1 = a


    if opt == 'Rectangular periódica':
        a = st.sidebar.number_input("Digite la amplitud de la señal: ")
        n = st.sidebar.number_input("Digite el número de armónicos: ")
        p = st.sidebar.number_input("Ingrese el periodo: ")
        if p>0:
            n = n+1
            wo = (2*np.pi)/p
            dt = 0.0001
            t = np.arange(0,int(p),dt)
            y = a*signal.square(wo*t, 0.5)

            graph.plot(t,y)
            plt.xlabel("t(s)")
            plt.ylabel("x(t)")
            plt.title("Gráfica x(t)")
            max1 = a

    if opt == 'Rampa trapezoidal':
        n = st.sidebar.number_input("Digite el número de armónicos: ")
        p = st.sidebar.number_input("Ingrese el periodo: ")
        if p>0:
            n = n+1
            wo = (2*np.pi)/p
            dt = 0.0001
            t = np.arange(0,int(p),dt)
            xinicio = 0
            xfinal = p
            if xfinal>xinicio:
                e=(xfinal-xinicio)/3
                def tramo1(z):         
                    return z-xinicio    
                def tramo2(z):         
                    return e    
                def tramo3(z):         
                    return -z+xfinal     
                a=xinicio
                b=xinicio+e   
                c=xinicio+2*e
                d=xinicio+3*e
                
                y=np.piecewise(t,[(a<=t) & (t<b),(b<=t)&(t<=c),(c<t)&(t<=d)],[lambda t:tramo1(t),lambda t: tramo2(t),lambda t:tramo3(t)])    
                tramo1=np.vectorize(tramo1)     
                graph.plot(t[t<b],tramo1(t[t<b]),c="c")  
                tramo2=np.vectorize(tramo2) 
                graph.plot(t[(b<=t)&(t<c)],tramo2(t[(b<=t)&(t<c)]),c="c") 
                tramo3=np.vectorize(tramo3)     
                graph.plot(t[(c<=t)&(t<=d)],tramo3(t[(c<=t)&(t<=d)]),c="c")  
                plt.xlabel("t(s)")
                plt.ylabel("x(t)")
                plt.title("Gráfica x(t)")
                max1 = e

    if opt == 'Exponencial':
        b = st.sidebar.number_input("Digite el factor de decrecimiento: ")
        n = st.sidebar.number_input("Digite el número de armónicos: ")
        p = st.sidebar.number_input("Ingrese el periodo: ")
        if p>0:
            n = n+1
            wo = (2*np.pi)/p
            dt = 0.0001
            t = np.arange(0,int(p),dt)
            y = np.exp(b*t)
            max1 = max(y)

            graph.plot(t,y)
            plt.xlabel("t(s)")
            plt.ylabel("x(t)")
            plt.title("Gráfica x(t)")

    if opt == 'Sinusoidal rectificada':
        a = st.sidebar.number_input("Digite la amplitud de la señal: ")
        n = st.sidebar.number_input("Digite el número de armónicos: ")
        p = st.sidebar.number_input("Ingrese el periodo: ")
        if p>0:
            n = n+1
            wo = (2*np.pi)/p
            dt = 0.0001
            t = np.arange(0,int(p),dt)
            y = a*abs(np.sin(wo*t))
            max1 = max(y)

            graph.plot(t,y)
            plt.xlabel("t(s)")
            plt.ylabel("x(t)")
            plt.title("Gráfica x(t)")

    st_a.pyplot(plt)

    if boton:
        #series de fourier
        #cálculo del coeficiente A0 (valor medio)
        ak = np.zeros((int(n),1))
        bk = np.zeros((int(n),1))
        m = np.size(t)

        A0 = 0
        for i in range(1,m):
            A0 = A0+(1/int(p))*y[i]*dt

        maxtotal1 = 0
        n_wo = np.arange(1,n,1)
        ck = np.zeros((int(n)))
        phik = np.zeros((int(n)))

        #cálculo de los coeficientes y espectros de fase y amplitud
        for i  in range(1,int(n)):
            for j in range(1,int(m)):
                ak[i] = ak[i]+((2/p)*y[j]*np.cos(i*t[j]*wo))*dt
                bk[i] = bk[i]+((2/p)*y[j]*np.sin(i*t[j]*wo))*dt
            ck[i] = (((ak[i])**2) + ((bk[i])**2))**(1/2)
            phik[i] = np.arctan((bk[i])/(ak[i]))*(-1)
            maxtotal=ck[i]
            maxtotal1=maxtotal1+maxtotal

        t2 = np.arange(0,2*p,dt)
        xf = 0*t2+A0

        fig2, (ax1, ax2) = plt.subplots(2, figsize=(8,8))
        st_b = st.pyplot(plt)

        for i in range(1,int(n)):
            xf = xf+ak[i]*np.cos(i*wo*t2)+bk[i]*np.sin(i*wo*t2)
            max2 = max(xf)
            max3 = max2/max1
            mm = xf/max3
            ax1.clear()
            ax2.clear()
            ax1.plot(t,y)
            ax2.plot(t2,mm)
            ax1.title.set_text('Señal original')
            ax2.title.set_text('Representación en series de Fourier')
            ax1.set(xlabel='t(s)', ylabel='x(t)')
            ax2.set(xlabel='t(s)', ylabel='x(t)')
            fig2.tight_layout()
            st_b.pyplot(fig2)

        st.write(maxtotal1)
        ck_1 = len(ck)
        phik_1 = len(phik)

        fig3, (ax_a, ax_b) = plt.subplots(2, figsize=(8,8))
        ax_a.title.set_text('Espectro de amplitud')
        ax_b.title.set_text('Espectro de fase')
        ax_a.stem(n_wo,ck[1:ck_1])
        ax_a.set(xlabel='Armónicos [N]', ylabel='Amplitud')
        ax_b.set(xlabel='Armónicos [N]', ylabel='Fase')
        ax_b.stem(n_wo,phik[1:phik_1])
        fig3.tight_layout()
        st_c = st.pyplot(plt)



if opt0 == 'Transformada de Fourier':
    st.title("Transformada de Fourier")
    st.latex("TF[x(t)] = \sum_{-∞}^{∞} x(t)*e⁻ᴶ*ʷ*ᵗ")
    st.caption("Transformación matemática empleada para transformar señales entre el dominio del tiempo y el dominio de la frecuencia. [1, secc. 4]")
    st.write("Se va a representar la siguiente función:")
    st.latex("x = A*sin(w_{o1}*n/fs)+A_{2}*cos(w_{o2}*n/fs)+A_{3}*sin(w_{o3}*n/fs)")
    fs = st.sidebar.number_input("Digite la frecuencia de muestreo")
    N = st.sidebar.number_input('Digite el numero de muestra')

    col1, col2, col3 = st.columns(3)
    with col1:
        a1 = st.number_input('Digite la primera amplitud [A]')
        wo1 = st.number_input('Digite la primera frecuencia [H𝓏]')

    with col2:
        a2 = st.number_input('Digite la segunda amplitud [A₂]')
        wo2 = st.number_input('Digite la segunda frecuencia [H𝓏]')

    with col3:
        a3 = st.number_input('Digite la tercera amplitud [A₃]:')
        wo3 = st.number_input('Digite la tercera frecuncia [H𝓏]')

    if fs >= 2*wo1 and fs >= 2*wo2 and fs >= 2*wo3:
        if fs > 0:
            n=np.arange(0,N)

            xt = a1*np.sin(2*np.pi*wo1*n/fs) + a2*np.cos(2*np.pi*wo2*n/fs) + a3*np.sin(2*np.pi*wo3*n/fs)
            y = np.fft.fft(xt)
            f1 = np.fft.fftfreq(y.size)*fs

            fig4, (graph1, graph2) = plt.subplots(2,1, figsize=(8,8))
            graph1.plot(n,xt)
            graph2.plot(f1,abs(y))
            graph1.title.set_text('Dominio del tiempo')
            graph2.title.set_text('Dominio de frecuencia')
            graph1.set(xlabel='Número de muestras [N]', ylabel='Amplitud')
            graph2.set(xlabel='Frecuencia [Hz]', ylabel='Amplitud')
            fig4.tight_layout()

            st_d = st.pyplot(plt)
    else:
        st.markdown("[!] Digite una frecuencia de muestreo por lo menos 2 veces mayor que la frecuencia de la señal")



with st.sidebar.expander("Bibliografía"):
    st.caption("[1] Portillo, J. P. T. (2017). Introducción a las señales y sistemas (1st ed.). Editorial Universidad del Norte. https://doi.org/10.2307/j.ctt1w6tf99")
