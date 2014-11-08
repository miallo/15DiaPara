# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/kevin.luedemann/.spyder2/.temp.py
"""

import maabara as ma
#import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import math as mt

#daten holen
data = np.genfromtxt("daten2.dat", delimiter="\t")
data = data*[1, 0.1]
Tantal_G=[1.0367,1.0372,1.0374]#g
Tantal_M=[1.0383,1.0379,1.0381]
Tantal_hoehe=[42.7]#mm
Tantal_h=np.mean(Tantal_hoehe)
Tantal_hsig=np.std(Tantal_hoehe)
Tantal_m=0.911#g
Tantal_p=16.6#g*cm^-3
Manganoxid_G=[0.5932,0.5925,0.5922]
Manganoxid_M=[0.6010,0.6013,0.6015]
Manganoxid_hoehe=[49.4]
Manganoxid_h=np.mean(Manganoxid_hoehe)
Manganoxid_hsig=np.std(Manganoxid_hoehe)
Manganoxid_m=0.455#g
Manganoxid_p=5#g*cm^-3
Wismut_G=[0.9493,0.9489,0.9503]
Wismut_M=[0.9476,0.9488,0.9491]
Wismut_hoehe=[21]
Wismut_h=np.mean(Wismut_hoehe)
Wismut_hsig=np.std(Wismut_hoehe)
Wismut_m=0.851#g
Wismut_p=9.8#g*cm^-3
#                  I=0, I=1.4, I=1.2, I=1.0, I=0.8
strom=np.array([[0.885, 0.889, 0.886, 0.886, 0.885],
                [0.886, 0.888, 0.887, 0.887, 0.886],
                [0.885, 0.887, 0.888, 0.886, 0.887]])
#               x   , I=1.4 I=1.2 I=1.0 I=0.8
feld=np.array([[33, 0.166, 0.134, 0.104, 0.068],
               [38, 0.155, 0.126, 0.096, 0.061],
               [43, 0.145, 0.120, 0.089, 0.056],
               [48, 0.136, 0.111, 0.082, 0.050],
               [53, 0.127, 0.103, 0.075, 0.045]])
                
wicklungen=3220
mue=4*mt.pi*10**(-7)
gerd=-9.81

#Auswertung 1
plt.xlabel('Position X [mm]')
plt.ylabel('Flussdichte B [T]')
plt.errorbar(data[:,0],data[:,1],yerr=0.01,xerr=0.1,label='Messwerte', fmt='g.')

#Auswertung 2
grad=np.gradient(data[:,1])
plt.plot(data[:,0], grad, 'r.', label='Gradient')
plt.legend(shadow=True, fancybox=True)
plt.savefig('Aus1.pdf', format='pdf')
print ("Werte für Tantal am Ort " + str(Tantal_h) + " \pm " + str(Tantal_hsig))
print ("B= " + str(data[16][1]) + ", dB= " + str(grad[16]))
Tantal_b=data[10][1]
Tantal_db=grad[10]
print ("Werte für Manganoxid am Ort " + str(Manganoxid_h) + " \pm " + str(Manganoxid_hsig))
print ("B= " + str(data[11][1]) + ", dB= " + str(grad[11]))
Manganoxid_b=data[11][1]
Manganoxid_db=grad[11]
print ("Werte für Tantal am Ort " + str(Wismut_h) + " \pm " + str(Wismut_hsig))
print ("B= " + str(data[5][1]) + ", dB= " + str(grad[5]))
Wismut_b=data[5]
Wismut_db=grad[5]

#Auswertung 3
plt.clf()
plt.xlabel('Position X [mm]')
plt.ylabel('Flussdichte B [T]')
produkt=grad*data[:,1]
plt.plot(data[:,0], produkt, 'r.', label='Produkt B und dB')
plt.legend(shadow=True, fancybox=True)
plt.savefig('Aus3.pdf', format='pdf')

#Auswertung 4 und 5 da 4 schwachsinn
#Tantal
dm=np.array(Tantal_G)-np.array(Tantal_M)
deltam=np.mean(dm)
print deltam
deltamsig=np.std(dm)
chi = ma.uncertainty.Sheet('-g_m*m_d*p/(B*m_0)')
chi.set_value('g_m', gerd*mue)
chi.set_value('m_d', deltam)
chi.set_error('m_d', deltamsig)
chi.set_value('p', Tantal_p)
chi.set_value('m_0', Tantal_m)
chi.set_value('B', Tantal_db)
chi.print_result('short', 'dot')
Tantal_chi=7.6*10**(-5)
Tantal_chisig=1.3*10**(-5)
Tantal_chip=Tantal_chi/Tantal_p
Tantal_chipsig=Tantal_chisig/Tantal_p
print str(Tantal_chip) + " \pm " + str(Tantal_chipsig)

#Manganoxid
dm=np.array(Manganoxid_G)-np.array(Manganoxid_M)
deltam=np.mean(dm)
print deltam
deltamsig=np.std(dm)
chi.set_value('p', Manganoxid_p)
chi.set_value('m_0', Manganoxid_m)
chi.set_value('B', Manganoxid_db)
chi.print_result('short', 'dot')
Manganoxid_chi=3.3*10**(-5)
Manganoxid_chisig=0.6*10**(-5)
Manganoxid_chip=Manganoxid_chi/Manganoxid_p
Manganoxid_chipsig=Manganoxid_chisig/Manganoxid_p
print (str(Manganoxid_chip) + " \pm " + str(Manganoxid_chipsig))

#Wismut
dm=np.array(Wismut_G)-np.array(Wismut_M)
deltam=np.mean(dm)
print deltam
deltamsig=np.std(dm)
chi.set_value('p', Wismut_p)
chi.set_value('m_0', Wismut_m)
chi.set_value('B', Wismut_db)
chi.print_result('short', 'dot')
Wismut_chi=4.7*10**(-5)
Wismut_chisig=0.8*10**(-5)
Wismut_chip=Wismut_chi/Wismut_p
Wismut_chipsig=Wismut_chisig/Wismut_p
print (str(Wismut_chip) + " \pm " + str(Wismut_chipsig))

#Auswertung 6
#Bild B-Feld
plt.clf()
plt.errorbar(feld[:,0], feld[:,4], fmt='r.', label='I=0.8 A', yerr=0.01, xerr=0.01)
plt.errorbar(feld[:,0], feld[:,3], fmt='g.', label='I=1.0 A', yerr=0.01, xerr=0.01)
plt.errorbar(feld[:,0], feld[:,2], fmt='b.', label='I=1.2 A', yerr=0.01, xerr=0.01)
plt.errorbar(feld[:,0], feld[:,1], fmt='m.', label='I=1.4 A', yerr=0.01, xerr=0.01)
plt.xlabel('Position X [mm]')
plt.ylabel('Flussdichte B [T]')
plt.legend(shadow=True, fancybox=True)
plt.savefig('Aus61.pdf', format='pdf')

#Bild Grad B-Feld
feld08=np.gradient(feld[:,4])
feld10=np.gradient(feld[:,3])
feld12=np.gradient(feld[:,2])
feld14=np.gradient(feld[:,1])
plt.clf()
plt.plot(feld[:,0], feld08, 'r.', label='I=0.8 A')
plt.plot(feld[:,0], feld10, 'g.', label='I=1.0 A')
plt.plot(feld[:,0], feld12, 'b.', label='I=1.2 A')
plt.plot(feld[:,0], feld14, 'm.', label='I=1.4 A')
plt.xlabel('Position X [mm]')
plt.ylabel('Flussdichte B\' [T]')
plt.legend(shadow=True, fancybox=True)
plt.savefig('Aus62.pdf', format='pdf')

#Auswertung 7
"""
db08=feld[2,4]*feld08[2]
db10=feld[2,3]*feld10[2]
db12=feld[2,2]*feld12[2]
db14=feld[2,1]*feld14[2]
Stromi=[0.8, 1.0, 1.2, 1.4]
force=np.array([])
forcesig=np.array([])
wert=db08*Tantal_chi*Tantal_m*Tantal_p/mue
np.append(force, np.array(wert))
wert=Tantal_chisig*db08*Tantal_m*Tantal_p/mue
np.append(forcesig, np.array(wert))
wert=db10*Tantal_chi*Tantal_m*Tantal_p/mue
np.append(force, np.array(wert))
wert=Tantal_chisig*db10*Tantal_m*Tantal_p/mue
np.append(forcesig, np.array(wert))
wert=db12*Tantal_chi*Tantal_m*Tantal_p/mue
np.append(force, np.array(wert))
wert=Tantal_chisig*db12*Tantal_m*Tantal_p/mue
np.append(forcesig, np.array(wert))
wert=db14*Tantal_chi*Tantal_m*Tantal_p/mue
np.append(force, np.array(wert))
wert=Tantal_chisig*db14*Tantal_m*Tantal_p/mue
np.append(forcesig, np.array(wert))
"""
Stromi=[0.8, 1.0, 1.2, 1.4]
f0=np.mean(strom[:,0])
f0sig=np.std(strom[:,0])
f08=np.mean(strom[:,4])
f08sig=np.std(strom[:,4])
f10=np.mean(strom[:,3])
f10sig=np.std(strom[:,3])
f12=np.mean(strom[:,2])
f12sig=np.std(strom[:,2])
f14=np.mean(strom[:,1])
f14sig=np.std(strom[:,1])
force=np.array([])
forcesig=np.array([])
np.append(np.array((f0-f08)*gerd), force)
force = np.append(force, np.array((f0-f08)*gerd))
force = np.append(force, np.array((f0-f10)*gerd))
force = np.append(force, np.array((f0-f12)*gerd))
force = np.append(force, np.array((f0-f14)*gerd))
forcesig = np.append(forcesig, np.array(mt.sqrt(f0sig**2*gerd**2/10000000+f08**2*gerd**2/10000000)))
forcesig = np.append(forcesig, np.array(mt.sqrt(f0sig**2*gerd**2/10000000+f10**2*gerd**2/10000000)))
forcesig = np.append(forcesig, np.array(mt.sqrt(f0sig**2*gerd**2/10000000+f12**2*gerd**2/10000000)))
forcesig = np.append(forcesig, np.array(mt.sqrt(f0sig**2*gerd**2/10000000+f14**2*gerd**2/10000000)))
m,b,tex=ma.linear_fit(Stromi,force,forcesig)

plt.clf()
plt.xlim(0.7,1.5)
plt.errorbar(Stromi, force, yerr=forcesig, fmt='r.', label='F(I)')
x=np.linspace(0.7,1.5)
tex=tex.replace('- -','-')
plt.plot( x, m.n*x+b.n, 'r-', label="$"+tex+"$")
plt.xlabel('Strom I [A]')
plt.ylabel('Kraft F [N]')
plt.legend(shadow=True, fancybox=True, loc=2)
plt.savefig('Aus7.pdf', format='pdf')
