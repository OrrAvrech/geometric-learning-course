import numpy as np


def get_astroid(a=5, start=-np.pi, stop=np.pi, n_samples=1000):
    t = np.linspace(start, stop, n_samples)
    x = a * np.cos(t)**3
    y = a * np.sin(t)**3
    curve = np.array([x, y])
    return curve


def get_folium_of_descrates(a=1, start=-100, stop=100, n_samples=1000):
    t = np.linspace(start, stop, n_samples)
    x = (3 * a * t) / (1 + t**3)
    y = (3 * a * t**2) / (1 + t**3)
    curve = np.array([x, y])
    return curve


def get_talbots(a=4, b=2, f=np.sqrt(12), start=-5, stop=5, n_samples=1000):
    t = np.linspace(start, stop, n_samples)
    x = (a**2 + (f**2) * np.sin(t)**2) * np.cos(t) / a
    y = (a**2 - 2*f**2 + (f**2) * np.sin(t)**2) * np.sin(t) / b
    curve = np.array([x, y])
    return curve


def get_witch_of_agensi(a=5, start=-5, stop=5, n_samples=1000):
    t = np.linspace(start, stop, n_samples)
    x = a * t
    y = a / (1 + t**2)
    curve = np.array([x, y])
    return curve


def get_cardioid(a=5, start=-np.pi, stop=np.pi, n_samples=1000):
    t = np.linspace(start, stop, n_samples)
    x = a * (2 * np.cos(t) - np.cos(2*t))
    y = a * (2 * np.sin(t) - np.sin(2*t))
    curve = np.array([x, y])
    return curve
