import streamlit as st
from PIL import Image
import numpy as np
import math
import time

class EulerCauchy:
    def __init__(self, coefficients=np.zeros(3)):
        self.coefficients = coefficients
    
    def m(self): # homogenous solution:
        m_poly = np.poly1d(0)
        for i, coefficient in enumerate(self.coefficients):
            poly = 1
            for i in range(0,len(self.coefficients)-1-i):
                poly *= np.poly1d([1,-i])
            m_poly += poly * coefficient
        return np.roots(m_poly)
        
    def roots(self):
        roots = {} # elements in form of {"m":i} where i is the number of repetitions of the root
        m_vals = self.m()
        for m_val in m_vals:
            matches = [root for root in roots.keys() if approx_eq(root, m_val) or approx_eq(root, np.conjugate(m_val))]
            if matches:
                roots[matches[0]] += 1
            else:
                roots[m_val] = 1
        return roots

    def y(self, x):
        y = 0
        i = 0
        roots = self.roots()
        for root in roots.keys():
            if root.imag:
                for n_rep in range(0, roots[root], 2):
                    y += (math.log(x) ** n_rep) * (x ** root.real) * (self.constants[i] * math.cos(root.imag * math.log(x)) 
                                                                      + self.constants[i+1] * math.sin(root.imag * math.log(x)))
                    i += 2
            else:
                for n_rep in range(roots[root]):
                    y += self.constants[i] * (math.log(x) ** n_rep) * (x ** root)
                    i += 1
        return float(y)
    
def approx_eq(a, b):
    return abs(a - b) < .0001
def normalize(values):
    return (values - np.min(values)) / (np.max(values) - np.min(values)) * 255
def valuemap(eq):
    return np.array(list(map(eq.y, np.arange(1,181))))
def generate_image(rx, gx, bx, ry, gy, by):
    rx = normalize(sum(valuemap(_) for _ in rx))
    gx = normalize(sum(valuemap(_) for _ in gx))
    bx = normalize(sum(valuemap(_) for _ in bx))
    ry = normalize(sum(valuemap(_) for _ in ry))
    gy = normalize(sum(valuemap(_) for _ in gy))
    by = normalize(sum(valuemap(_) for _ in by))

    x_values = np.stack([rx, gx, bx]).T
    y_values = np.stack([ry, gy, by]).T
    
    x_values = np.repeat(x_values[:,np.newaxis,:], 180, axis=1)
    y_values = np.repeat(y_values[np.newaxis,:,:], 180, axis=0)
    return ((x_values + y_values) / 2).astype(np.uint8)

def display_solution(eq, constants, ind, dep):
    solution = f"{dep} = "
    i = 0
    roots = eq.roots()
    for root in roots.keys():
        if root.imag:
            for n_rep in range(0, roots[root], 2):
                match n_rep:
                    case 0:
                        solution += f"+ {ind}^{{{round(root.real, 2)}}} ({constants[i]}\cos ({round(root.imag, 2)}\ln {ind}) + {constants[i+1]}\sin ({round(root.imag, 2)}\ln {ind}))"
                    case 1:
                        solution += f"+ {ind}^{{{round(root.real, 2)}}} \ln {ind} ({constants[i]}\cos ({round(root.imag, 2)}\ln {ind}) + {constants[i+1]}\sin ({round(root.imag, 2)}\ln {ind}))"
                    case _:
                        solution += f"+ {ind}^{{{round(root.real, 2)}}} (\ln {ind})^{n_rep} ({constants[i]}\cos ({round(root.imag, 2)}\ln {ind}) + {constants[i+1]}\sin ({round(root.imag, 2)}\ln {ind}))"
                i += 2
        else:
            for n_rep in range(roots[root]):
                match n_rep:
                    case 0:
                        solution += f"+ {constants[i]}{ind}^{{{round(root.real, 2)}}}"
                    case 1:
                        solution += f"+ {constants[i]}{ind}^{{{round(root.real, 2)}}} \ln {ind}"
                    case _:
                        solution += f"+ {constants[i]}{ind}^{{{round(root.real, 2)}}} (\ln {ind})^{n_rep}"
                i += 1
    return solution.replace("= +", "=")

st.set_page_config(layout="wide")

dependents = ('R', 'G', 'B')
independents = ('x', 'y')

config, gen = st.tabs(['Configuration','Generation'])
with config:
    options, image = st.columns(2)
    with options:
        eqs = {f'{dep}{ind}':(EulerCauchy(np.zeros(3)),) for dep in dependents for ind in independents}
        for ind in independents:
            for dep in dependents:
                expander = st.expander(f"#### {dep}({ind})")
                with expander:
                    sliders, equation = st.columns(2)
                    with sliders:
                        order = st.slider("Order of Equation", min_value=2, max_value=5, key=f"{dep}{ind}")
                        display_eq = ""
                        for j in range(order + 1, 0, -1):
                            display_eq += "{}" + (f"{ind}^{j-1}" if j-1 > 1 else ind if j-1 == 1 else "") + dep + '\'' * (j - 1) + " + "
                        display_eq += "= 0"
                        coefficients = tuple(st.slider(f"ODE Coefficient {j + 1}", 
                                                            value=.5, min_value=0.0, max_value=10.0, step=.01, key=f"ODE {dep}{ind}{j}") 
                                                            for j in range(order + 1))
                    
                    with equation:
                        st.write("##### ODE")
                        st.latex(display_eq.format(*coefficients).replace("+ -", "- ").replace("+ =", "="))
                        st.write("##### Solution")
                        constants = np.ones(order)
                        eq = EulerCauchy(coefficients)
                        eq.constants = constants
                        st.latex(display_solution(eq, constants, ind, dep))
                        constants = tuple(st.slider(f"Solution Coefficient {j + 1}", 
                                                        min_value=0.0, max_value=1.0, value=1.0, step=.01, key=f"SOL {dep}{ind}{j}") 
                                                        for j in range(order))
                        eq = EulerCauchy(coefficients)
                        eq.constants = constants
                        eqs[f'{dep}{ind}'] = (eq,)

    with gen:
        max_time = 5 * 60
        test_image = st.file_uploader("Upload a gradient image", type=['png', 'jpg', 'jpeg'])
        if test_image:
            time_bar = st.progress(0, text="Time elapsed")
            error_bar = st.progress(0, text="Mean squared error")

            image = Image.open(test_image)
            image = image.resize((180, 180))
            image = image.crop((0,0,180,180))
            image = np.asarray(image, dtype=float)
            
            og, generated = st.columns(2)

            with og:
                st.image(image/255, use_column_width=True)
            lowest_mse = None
            start_time = time.time()

            with generated:
                generated_image_display = st.empty()
                error_display = st.empty()
            
            while True:
                constants = tuple([
                    {
                        'coefficients': np.random.rand(3) * 10,
                        'constants': np.random.rand(3),
                    } for _ in range(10)
                ] for _ in range(6))
                
                axes = []
                for axis in constants:
                    eqs = []
                    for constant in axis:
                        eq = EulerCauchy(constant['coefficients'])
                        eq.constants = constant['constants']
                        eqs.append(eq)
                    axes.append(tuple(eqs))

                generated_image = generate_image(*tuple(axes))
                mse = np.sum((image - generated_image) ** 2) / image.size
                if not lowest_mse:
                    lowest_mse = mse
                    max_error = mse
                elif mse < lowest_mse:
                    error_bar.progress(lowest_mse / max_error, text="Mean squared error")
                    lowest_mse = mse
                    generated_image_display.image(generated_image, use_column_width=True)
                    error_display.write(f"Mean squared error: {int(mse)}")
                
                if time.time() - start_time > max_time:
                    break
                time_bar.progress((time.time() - start_time)/max_time, text="Time elapsed")