def a():
    f = 2000  # Frecventa de esantionare (Hz)
    T = 1 / f  # Intervalul de timp dintre esantioane (secunde)
    print(f'Intervalul de timp dintre esantioane este {T} secunde.')


def b():
    f = 2000  # Frecventa de esantionare (Hz)
    h = 3600  # Secunde pe ora
    dim_esantion = 4  # Dimensiunea esantionului (biti)
    byte = 8  # Biti per byte
    bits_h = f * h * dim_esantion  # Numarul de biti necesari pentru o ora
    bytes_h = bits_h / byte  # Numarul de bytes necesari pentru o ora
    print(
        f'Numarul de bytes necesari este {bytes_h}.')


if __name__ == '__main__':
    a()
    b()
