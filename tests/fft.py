import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.fftpack import dct, idct

L = 85
t = np.arange(0, 340, 1)
t = t / 2
x = np.exp(t/400) + np.sin(2*np.pi*t/17) + 0.5*np.sin(2*np.pi*t/10) + np.random.randn(340)
#!
N = 340
fourier = np.abs(np.fft.fft(x, n=N))
print(np.sum(fourier ** 2) / len(x))
energy = np.linalg.norm(x, ord=2) ** 2

ff = fourier[0:N//2+1]
# print(np.abs(ff[1:170])-np.abs(fourier[171:][::-1]))
ff[1:N//2] = np.sqrt((ff[1:N//2]**2 + fourier[N//2+1:][::-1]**2))

f, a = periodogram(x)

print(energy, np.sum(ff**2)/N)

print(energy/len(x), np.sum((ff / np.sqrt(N*len(x)))**2))

plt.figure()
plt.plot(a)
plt.plot(ff)
plt.show()


