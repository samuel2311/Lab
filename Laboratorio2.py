from typing import final
import streamlit as st
import math
from re import X
from turtle import end_fill
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import integrate
from scipy.integrate import simps
#import sk_dsp_comm.sigsys as ss
import time

st.sidebar.title("Lab 2: Signal modelation")
st.title('Convolutions')
st.subheader('Mathematical operation of 2 functions to create a new one')

st.sidebar.header("Signal generation")

dom = st.sidebar.selectbox('Please select the domain of the function', ['-', 'Discrete domain', 'Continuos domain'])
if dom == 'Discrete domain':
     dom = 1
else:
     dom = 0

option = st.sidebar.selectbox(
     'Select your first function, X(t)',
     ('-', 'Exponential', 'Sinusoidal', 'Triangular', 'Rectangular', 'Ramp type 1', 'Ramp type 2', 'Ramp type 3'))

st.sidebar.write('You selected', option, 'as your h(t)')

fig, (graph1) = plt.subplots(1,1)
fila_graficas=st.columns(2)
with fila_graficas[0]:
    graph = st.pyplot(plt)
s=0.001

if option == 'Sinusoidal':
     st.sidebar.latex("x(t) = A*sin(2*pi*f*t)")
     a = st.sidebar.number_input('Please type the value for amplittude', key = 1)
     f = st.sidebar.number_input('Please type the value for the frequency', key = 2)
     xinicio = st.sidebar.number_input('Type the point of start for the function', key = 3)
      
     if f == 0:
          t = 0
          xfinal = 0
     else:
          xfinal=(1/f)+xinicio
          t=np.arange(xinicio,xfinal+s,s)
          if dom== 1:
               s=(1/(20*f))
               t=np.arange(xinicio,xfinal,s)
               vx=np.arange(((xfinal-xinicio)/s))
               i=0
               pos=0
               while i<abs(xfinal) and pos<abs(((xfinal-xinicio)/s)):
                    vx[pos]=a*np.sin(2*np.pi*f*i)
                    i=i+s
                    pos=pos+1
               with fila_graficas[0]:
                    plt.xlabel("t(s)")
                    plt.ylabel("x(t)")
                    graph1.stem(t,vx)
          else:
               y=a*np.sin(2*np.pi*f*t)
               plt.xlabel("t(s)")
               plt.ylabel("x(t)")
               graph1.plot(t,y)

if option == 'Exponential':
     st.sidebar.latex("x(t) = A*e^. (B*t)")
     a = st.sidebar.number_input('Please type a value for A', key = 1)
     b = st.sidebar.number_input('Please type a value for B', key = 2)
     t=np.arange(0,5,s)
     if dom == 1:
          xinicio = st.sidebar.number_input('Please type the point of start for the function', key = 3)
          xfinal = st.sidebar.number_input('Please type the end point for the function', key='4' )
          s = 0.1
          t = np.arange(xinicio, xfinal,s)
          vx = np.arange((xfinal-xinicio)/s)
          i = 0
          pos = 0
          while i<abs(xfinal) and pos<abs((xfinal-xinicio)/s):
               vx[pos] = a*np.exp(i)
               i=i+s
               pos=pos+1
          with fila_graficas[0]:
               plt.xlabel("t(s)")
               plt.ylabel("x(t)")
               graph1.stem(t,vx)
     else:
          y = a*np.exp(b*t)
          plt.xlabel("t(s)")
          plt.ylabel("x(t)")
          graph1.plot(t,y)

if option == 'Triangular':          
     st.sidebar.latex("Triangular")
     a = st.sidebar.number_input('Please type the value for the amplittude')
     f = st.sidebar.number_input('Please type the value for the frequency')
     xinicio = st.sidebar.number_input('Please type the start point for the frequency')
     if f == 0 :
        t=0
        xfinal=0
     else :
          xfinal = 3/f+xinicio
          t= np.arange (xinicio, xfinal+s,s)
          if dom==1 :
               s=1/(10*f)
               t = np.arange (xinicio, 3/f+xinicio,s)
               vx=np.arange((xfinal-xinicio)/s)
               i=0
               pos=0
               while i<abs(xfinal) and pos<abs((xfinal-xinicio)/s):
                    vx[pos]=a*signal.sawtooth(2*np.pi*f*i,0.5)
                    i=i+s
                    pos=pos+1
               with fila_graficas[0]:
                    plt.xlabel("t(s)")
                    plt.ylabel("x(t)")
                    graph1.stem(t,vx)
          else:
               t = np.arange (xinicio, xfinal+s,s)
               y = a*signal.sawtooth(2*np.pi*f*t,0.5)
               graph1.plot(t,y)
               with fila_graficas[0]:
                    plt.xlabel("t(s)")
                    plt.ylabel("x(t)")



if option == 'Rectangular':
     st.sidebar.latex("Rectangular")
     a = st.sidebar.number_input('Please specify the amplittude',key=1)
     f = st.sidebar.number_input('Please specify the frequency',key=2)
     xinicio = st.sidebar.number_input('Please specify the start point for the function',key=3)
     if f == 0:
          t=0
     else:
          xfinal = 2/f+xinicio
          t = np.arange(xinicio, xfinal+s, s)
          if dom == 1:
               s=1/(10*f)
               t = np.arange (xinicio, 2/f+xinicio,s)
               vx=np.arange((xfinal-xinicio)/s)
               i=0
               pos=0
               while i<abs(xfinal) and pos<abs((xfinal-xinicio)/s):
                    vx[pos]=signal.square(2*np.pi*i*f)
                    i=i+s
                    pos=pos+1
               with fila_graficas[0]:
                    graph1.stem(t,vx)
          else:
               y = signal.square(2*np.pi*f*t)
               graph1.plot(t,y)

if option == 'Ramp type 1':
          xinicio=st.sidebar.number_input("Please specify the start point for the function",key=1)
          xfinal=st.sidebar.number_input("Please specify the end point for the function",key=2)
          if xfinal>xinicio:
               b=((xfinal-xinicio)/3)*1
               c=((xfinal-xinicio)/3)*2

               def tramo1 (x1):
                    return xinicio
               def tramo2 (x1):
                return x1-b
               def tramo3 (x1):
                    return b
               if dom==1:
                    s=((xfinal-xinicio)/20)
                    t =np.arange(xinicio,xfinal,s) 
                    vx=np.linspace(1,20,20)
                    i=xinicio
                    pos=0
                    while i < xfinal and pos<((xfinal-xinicio)/s):
                         if i<b:
                              vx[pos]=tramo1(i)    
                         if i>b and i<c:
                              vx[pos]=tramo2(i)
                         if i>c:
                              vx[pos]=tramo3(i)
                         pos=pos+1
                         i=i+s
                    with fila_graficas[0]:
                         plt.xlabel("t(s)")
                         plt.ylabel("x(t)")
                         plt.title("Gráfica x(t)")
                         graph1.stem(t,vx)
               else:
                    x1=np.arange(xinicio,xfinal,0.01) 
                    y1=np.piecewise(x1,[(xinicio<=x1) & (x1<b), (b<=x1) & (x1<=c), (c<x1) & (x1<=xfinal)],[lambda x:tramo1(x1), lambda x1:tramo2(x1), lambda x1:tramo3(x1)])
                    tramo1=np.vectorize(tramo1)
                    tramo3=np.vectorize(tramo3)
                    plt.xlabel("t(s)")
                    plt.ylabel("x(t)")
                    graph1.plot(x1,y1)


if option == 'Ramp type 2':
     xinicio = st.sidebar.number_input('Please specify the start point', key = 1)
     xfinal = st.sidebar.number_input('Please specify the end point', key = 2)
     if xfinal>xinicio:
          e=(xfinal-xinicio)/2
          def tramo2(z):         
            return e    
          def tramo3(z):         
            return -z+xfinal    
          a=xinicio 
          b=xinicio+e  
          c=xinicio+2*e
          if dom==1:
               s=(xfinal-xinicio)/20
               t =np.arange(xinicio,xfinal,s) 
               vx=np.linspace(1,20,20)
               i=xinicio
               pos=0
               while i < xfinal and pos<((xfinal-xinicio)/s):    
                    if i>a and i<b:
                         vx[pos]=tramo2(i)
                    if i>b and i<c:
                         vx[pos]=tramo3(i)
                         pos=pos+1
                         i=i+s
                    with fila_graficas[0]:
                         plt.xlabel("t(s)")
                         plt.ylabel("x(t)")
                         plt.title("Gráfica x(t)")
                         graph1.stem(t,vx)
          else:
               t= np.arange(xinicio, xfinal+s, s)
               y=np.piecewise(t,[(a<=t)&(t<b), (b<=t)&(t<=c)],[lambda t: tramo2(t),lambda t:tramo3(t)])      
               tramo2=np.vectorize(tramo2) 
               graph1.plot(t[(a<=t)&(t<b)],tramo2(t[(a<=t)&(t<b)]),c="c") 
               tramo3=np.vectorize(tramo3)     
               graph1.plot(t[(b<=t)&(t<=c)],tramo3(t[(b<=t)&(t<=c)]),c="c")  
               with fila_graficas[0]:
                    plt.xlabel("t(s)")
                    plt.ylabel("x(t)")
                    plt.title("Gráfica x(t)")
                    
if option == 'Ramp type 3':
    xinicio = st.sidebar.number_input("Ingrese punto de inicio",key=1)
    xfinal = st.sidebar.number_input("Ingrese punto final",key=2)
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
        if dom==1:
            s=(xfinal-xinicio)/20
            t =np.arange(xinicio,xfinal,s) 
            vx=np.linspace(1,20,20)
            i=xinicio
            pos=0
            while i < xfinal and pos<((xfinal-xinicio)/s):
                if i<b:
                    vx[pos]=tramo1(i)    
                if i>b and i<c:
                    vx[pos]=tramo2(i)
                if i>c and i<d:
                    vx[pos]=tramo3(i)
                pos=pos+1
                i=i+s
            with fila_graficas[0]:
                    plt.xlabel("t(s)")
                    plt.ylabel("x(t)")
                    plt.title("Gráfica x(t)")
                    graph1.stem(t,vx)
        else:
            t= np.arange(xinicio, xfinal+s, s)
            y=np.piecewise(t,[(a<=t) & (t<b),(b<=t)&(t<=c),(c<t)&(t<=d)],[lambda t:tramo1(t),lambda t: tramo2(t),lambda t:tramo3(t)])    
            tramo1=np.vectorize(tramo1)     
            graph1.plot(t[t<b],tramo1(t[t<b]),c="c")  
            tramo2=np.vectorize(tramo2) 
            graph1.plot(t[(b<=t)&(t<c)],tramo2(t[(b<=t)&(t<c)]),c="c") 
            tramo3=np.vectorize(tramo3)     
            graph1.plot(t[(c<=t)&(t<=d)],tramo3(t[(c<=t)&(t<=d)]),c="c")  
            with fila_graficas[0]:
                    plt.xlabel("t(s)")
                    plt.ylabel("x(t)")
                    plt.title("Gráfica x(t)")

graph.pyplot(plt)

option2 = st.sidebar.selectbox(
     'Select your second function, h(t)',
     ['-', 'Exponential', 'Sinusoidal', 'Triangular', 'Rectangular', 'Ramp type 1', 'Ramp type 2', 'Ramp type 3'])

st.sidebar.write('You selected', option2, 'as your h(t)')
fig2, (graph2) = plt.subplots(nrows=1,ncols=1)
with fila_graficas[1]:
    graph22 = st.pyplot(plt)

if option2 == 'Sinusoidal':
     st.sidebar.latex("x(t) = A*sin(2*pi*f*t)")
     a2 = st.sidebar.number_input('Please type the value for amplittude', key = 4)
     f2 = st.sidebar.number_input('Please type the value for the frequency', key = 5)
     xinicio2 = st.sidebar.number_input('Type the point of start for the function', key = 6)
      
     if f2 == 0:
          t2 = 0
          xfinal2 = 0
     else:
          xfinal2=(1/f2)+xinicio2
          t2=np.arange(xinicio2,xfinal2+s,s)
          if dom== 1:
               s=(1/(20*f2))
               t2=np.arange(xinicio2,xfinal2,s)
               vx2=np.arange(((xfinal2-xinicio2)/s))
               i2=0
               pos2=0
               while i2<abs(xfinal2) and pos2<abs(((xfinal2-xinicio2)/s)):
                    vx2[pos2]=a2*np.sin(2*np.pi*f2*i2)
                    i2=i2+s
                    pos2=pos2+1
               with fila_graficas[1]:
                    plt.xlabel("t(s)")
                    plt.ylabel("x(t2)")
                    graph2.stem(t2,vx2)
          else:
               y2=a2*np.sin(2*np.pi*f2*t2)
               plt.xlabel("t(s)")
               plt.ylabel("x(t)")
               graph2.plot(t2,y2)

if option2 == 'Exponential':
     st.sidebar.latex("x(t) = A*e^. (B*t)")
     a = st.sidebar.number_input('Please type a value for A', key = 1)
     b = st.sidebar.number_input('Please type a value for B', key = 2)
     t2=np.arange(0,5,s)
     if dom == 1:
          xinicio2 = st.sidebar.number_input('Please type the point of start for the function', key = 3)
          xfinal2 = st.sidebar.number_input('Please type the end point for the function', key='4' )
          s = 0.1
          t2 = np.arange(xinicio2, xfinal2,s)
          vx2 = np.arange((xfinal2-xinicio2)/s)
          i2 = 0
          pos2 = 0
          while i2<abs(xfinal2) and pos2<abs((xfinal2-xinicio2)/s):
               vx2[pos2] = a*np.exp(i2)
               i2=i2+s
               pos2=pos2+1
          with fila_graficas[1]:
               plt.xlabel("t2(s)")
               plt.ylabel("x(t2)")
               graph1.stem(t2,vx2)
     else:
          y = a*np.exp(b*t2)
          plt.xlabel("t2(s)")
          plt.ylabel("x(t2)")
          graph1.plot(t2,y)

if option2 == 'Triangular':          
     st.sidebar.latex("sawtooth")
     a2 = st.sidebar.number_input('Please type the value for the amplittude', key =4)
     f2 = st.sidebar.number_input('Please type the value for the frequency', key=5)
     xinicio2 = st.sidebar.number_input('Please type the start point for the frequency',key=6)
     if f2 == 0 :
        t2=0
        xfinal2=0
     else :
          xfinal2 = 3/f2+xinicio2
          t2= np.arange (xinicio2, xfinal2+s,s)
          if dom==1 :
               s=1/(10*f2)
               t2 = np.arange (xinicio2, 3/f2+xinicio2,s)
               vx2=np.arange((xfinal2-xinicio2)/s)
               i2=0
               pos2=0
               while i2<abs(xfinal2) and pos2<abs((xfinal2-xinicio2)/s):
                    vx2[pos2]=a2*signal.sawtooth(2*np.pi*f2*i2,0.5)
                    i2=i2+s
                    pos2=pos2+1
               with fila_graficas[1]:
                    plt.xlabel("t2(s)")
                    plt.ylabel("x(t2)")
                    graph2.stem(t2,vx2)
          else:
               t2 = np.arange (xinicio2, xfinal2+s,s)
               y2 = a2*signal.sawtooth(2*np.pi*f2*t2,0.5)
               graph2.plot(t2,y2)
               with fila_graficas[1]:
                    plt.xlabel("t(s)")
                    plt.ylabel("x(t)")



if option2 == 'Rectangular':
     st.sidebar.latex("Rectangular")
     a = st.sidebar.number_input('Please specify the amplittude',key=4)
     f2 = st.sidebar.number_input('Please specify the frequency',key=5)
     xinicio2 = st.sidebar.number_input('Please specify the start point for the function',key=6)
     if f2 == 0:
          t2=0
     else:
          xfinal2 = 2/f2+xinicio2
          t2 = np.arange(xinicio2, xfinal2+s, s)
          if dom == 1:
               s=1/(10*f2)
               t2 = np.arange (xinicio2, 2/f2+xinicio2,s)
               vx2=np.arange((xfinal2-xinicio2)/s)
               i2=0
               pos2=0
               while i2<abs(xfinal2) and pos2<abs((xfinal2-xinicio2)/s):
                    vx2[pos2]=signal.square(2*np.pi*i2*f2)
                    i2=i2+s
                    pos2=pos2+1
               with fila_graficas[1]:
                    graph2.stem(t2,vx2)
          else:
               y2 = signal.square(2*np.pi*f2*t2)
               graph2.plot(t2,y2)

if option2 == 'Ramp type 1':
          xinicio2=st.sidebar.number_input("Please specify the start point for the function" ,key=4)
          xfinal2=st.sidebar.number_input("Please specify the end point for the function",key=5)
          if xfinal2>xinicio2:
               b=((xfinal2-xinicio2)/3)*1
               c=((xfinal2-xinicio2)/3)*2

               def tramo1 (x2):
                    return xinicio2
               def tramo2 (x2):
                return x2-b
               def tramo3 (x2):
                    return b
               if dom==1:
                    s=((xfinal2-xinicio2)/20)
                    t =np.arange(xinicio2,xfinal2,s) 
                    vx2=np.linspace(1,20,20)
                    i=xinicio2
                    pos=0
                    while i < xfinal2 and pos<((xfinal2-xinicio2)/s):
                         if i<b:
                              vx2[pos]=tramo1(i)    
                         if i>b and i<c:
                              vx2[pos]=tramo2(i)
                         if i>c:
                              vx2[pos]=tramo3(i)
                         pos=pos+1
                         i=i+s
                    with fila_graficas[0]:
                         plt.xlabel("t(s)")
                         plt.ylabel("x(t)")
                         plt.title("Gráfica x(t)")
                         graph2.stem(t,vx)
               else:
                    x2=np.arange(xinicio2,xfinal2,0.01) 
                    y2=np.piecewise(x2,[(xinicio2<=x2) & (x2<b), (b<=x2) & (x2<=c), (c<x2) & (x2<=xfinal2)],[lambda x:tramo1(x2), lambda x2:tramo2(x2), lambda x2:tramo3(x2)])
                    tramo1=np.vectorize(tramo1)
                    tramo3=np.vectorize(tramo3)
                    plt.xlabel("t(s)")
                    plt.ylabel("x(t)")
                    graph2.plot(x2,y2)

                    
if option2 == 'Ramp type 2':
     xinicio2 = st.sidebar.number_input('Please specify the start point', key = 4)
     xfinal2 = st.sidebar.number_input('Please specify the end point', key = 5)
     if xfinal2>xinicio2:
          e=(xfinal2-xinicio2)/2
          def tramo2(z):         
            return e    
          def tramo3(z):         
            return -z+xfinal    
          a=xinicio2 
          b=xinicio2+e  
          c=xinicio2+2*e
          if dom==1:
               s=(xfinal2-xinicio2)/20
               t =np.arange(xinicio2,xfinal2,s) 
               vx2=np.linspace(1,20,20)
               i=xinicio2
               pos=0
               while i < xfinal2 and pos<((xfinal2-xinicio2)/s):    
                    if i>a and i<b:
                         vx2[pos]=tramo2(i)
                    if i>b and i<c:
                         vx2[pos]=tramo3(i)
                         pos=pos+1
                         i=i+s
                    with fila_graficas[0]:
                         plt.xlabel("t(s)")
                         plt.ylabel("x(t)")
                         plt.title("Gráfica x(t)")
                         graph2.stem(t,vx2)
          else:
               t= np.arange(xinicio2, xfinal2+s, s)
               y2=np.piecewise(t,[(a<=t)&(t<b), (b<=t)&(t<=c)],[lambda t: tramo2(t),lambda t:tramo3(t)])      
               tramo2=np.vectorize(tramo2) 
               graph2.plot(t[(a<=t)&(t<b)],tramo2(t[(a<=t)&(t<b)]),c="c") 
               tramo3=np.vectorize(tramo3)     
               graph2.plot(t[(b<=t)&(t<=c)],tramo3(t[(b<=t)&(t<=c)]),c="c")  
               with fila_graficas[0]:
                    plt.xlabel("t(s)")
                    plt.ylabel("x(t)")
                    plt.title("Gráfica x(t)")
                    
if option2 == 'Ramp type 3':
    xinicio2 = st.sidebar.number_input("Ingrese punto de inicio",key=4)
    xfinal2 = st.sidebar.number_input("Ingrese punto final",key=5)
    if xfinal2>xinicio2:
        e=(xfinal2-xinicio2)/3
        def tramo1(z):         
            return z-xinicio2    
        def tramo2(z):         
            return e    
        def tramo3(z):         
            return -z+xfinal2     
        a=xinicio2
        b=xinicio2+e   
        c=xinicio2+2*e
        d=xinicio2+3*e
        if dom==1:
            s=(xfinal2-xinicio2)/20
            t =np.arange(xinicio2,xfinal2,s) 
            vx2=np.linspace(1,20,20)
            i=xinicio2
            pos=0
            while i < xfinal2 and pos<((xfinal2-xinicio2)/s):
                if i<b:
                    vx2[pos]=tramo1(i)    
                if i>b and i<c:
                    vx2[pos]=tramo2(i)
                if i>c and i<d:
                    vx2[pos]=tramo3(i)
                pos=pos+1
                i=i+s
            with fila_graficas[0]:
                    plt.xlabel("t(s)")
                    plt.ylabel("x(t)")
                    plt.title("Gráfica x(t)")
                    graph2.stem(t,vx2)
        else:
            t= np.arange(xinicio2, xfinal2+s, s)
            y2=np.piecewise(t,[(a<=t) & (t<b),(b<=t)&(t<=c),(c<t)&(t<=d)],[lambda t:tramo1(t),lambda t: tramo2(t),lambda t:tramo3(t)])    
            tramo1=np.vectorize(tramo1)     
            graph2.plot(t[t<b],tramo1(t[t<b]),c="c")  
            tramo2=np.vectorize(tramo2) 
            graph1.plot(t[(b<=t)&(t<c)],tramo2(t[(b<=t)&(t<c)]),c="c") 
            tramo3=np.vectorize(tramo3)     
            graph2.plot(t[(c<=t)&(t<=d)],tramo3(t[(c<=t)&(t<=d)]),c="c")  
            with fila_graficas[0]:
                    plt.xlabel("t(s)")
                    plt.ylabel("x(t)")
                    plt.title("Gráfica x(t)")


graph22.pyplot(plt) 

pos_boton=st.columns(7)
with pos_boton[3]:
    conv = st.button('Convolve')
if conv:
    fig, (graph3, graph4) = plt.subplots(2,1)
    st_a = st.pyplot(plt)
     #ty = np.arange(xinicio+xinicio2,xfinal+xfinal2+s, s)
if dom == 1:
     p1=[0.35,0.8]
     p2=[0.7,-0.2]
     a,b=-5,5

     #Function
     def g(x,p):
          sig=p[0]
          mu=p[1]
          return 1.0/(sig*np.sqrt(2.*np.pi)*np.exp(-1.*(x-mu)**2)/(2*sig**2))

     #Calcular convolución 
     x=np.linspace(0,xfinal2,100)
     conb=[]
     for xx in x:
          xp=np.linspace(0,xx,100)
          h=g(xp,vx)*g(xx-xp,vx2)
          I=simps(h,xp)
          conb.append(I)
     plt.plot(x,conb,label="Convolución")
     xr=np.linspace(a,b,1000)
     plt.plot(xr,g(xr,vx),label="Función 1")
     plt.plot(xr,g(xr,vx2),label="Función 2")
     plt.axvline(0,c='gray',ls='--',lw=1)
     plt.axhline(0,c='gray',ls='--',lw=1)
     plt.legend(loc='best', prop={"size":10})
     #plt.show()
     st_a.pyplot(plt)
else:     
     p1=[0.35,0.8]
     p2=[0.7,-0.2]
     a,b=-5,5

     #Function
     def g(x,p):
          sig=p[0]
          mu=p[1]
          return 1.0/(sig*np.sqrt(2.*np.pi)*np.exp(-1.*(x-mu)**2)/(2*sig**2))

     #Calcular convolución 
     x=np.linspace(0,xfinal2,100)
     conb=[]
     for xx in x:
          xp=np.linspace(0,xx,100)
          h=g(xp,y)*g(xx-xp,y2)
          I=simps(h,xp)
          conb.append(I)
     plt.plot(x,conb,label="Convolución")
     xr=np.linspace(0,xfinal2,1000)
     plt.plot(xr,g(xr,y),label="Función 1")
     plt.plot(xr,g(xr,y2),label="Función 2")
     plt.axvline(0,c='gray',ls='--',lw=1)
     plt.axhline(0,c='gray',ls='--',lw=1)
     plt.legend(loc='best', prop={"size":10})
     #plt.show()
     st_a.pyplot(plt)

