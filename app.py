import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import math

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
def display_solution(eq, constants):
    solution = "y = "
    i = 0
    roots = eq.roots()
    for root in roots.keys():
        if root.imag:
            for n_rep in range(0, roots[root], 2):
                match n_rep:
                    case 0:
                        solution += f"+ x^{{{round(root.real, 2)}}} ({constants[i]}\cos ({round(root.imag, 2)}\ln x) + {constants[i+1]}\sin ({round(root.imag, 2)}\ln x))"
                    case 1:
                        solution += f"+ x ^{{{round(root.real, 2)}}} \ln x ({constants[i]}\cos ({round(root.imag, 2)}\ln x) + {constants[i+1]}\sin ({round(root.imag, 2)}\ln x))"
                    case _:
                        solution += f"+ x^{{{round(root.real, 2)}}} (\ln x)^{n_rep} ({constants[i]}\cos ({round(root.imag, 2)}\ln x) + {constants[i+1]}\sin ({round(root.imag, 2)}\ln x))"
                i += 2
        else:
            for n_rep in range(roots[root]):
                match n_rep:
                    case 0:
                        solution += f"+ {constants[i]}x^{{{round(root.real, 2)}}}"
                    case 1:
                        solution += f"+ {constants[i]}x^{{{round(root.real, 2)}}} \ln x"
                    case _:
                        solution += f"+ {constants[i]}x^{{{round(root.real, 2)}}} (\ln x)^{n_rep}"
                i += 1
    return solution.replace("= +", "=")

st.set_page_config(layout="wide")

options, image = st.columns(2)
with options:
    dependents = ('R', 'G', 'B')
    independents = ('x', 'y')
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
                        display_eq += "{}" + (f"x^{j-1}" if j-1 > 1 else "x" if j-1 == 1 else "") + "y" + '\'' * (j - 1) + " + "
                    display_eq += "= 0"
                    coefficients = tuple(st.slider(f"ODE Coefficient {j + 1}", 
                                                        value=.5, min_value=0.0, max_value=1.0, step=.01, key=f"ODE {dep}{ind}{j}") 
                                                        for j in range(order + 1))
                
                with equation:
                    st.write("##### ODE")
                    st.latex(display_eq.format(*coefficients).replace("+ -", "- ").replace("+ =", "="))
                    st.write("##### Solution")
                    constants = tuple(st.slider(f"Solution Coefficient {j + 1}", 
                                                    min_value=0.0, max_value=1.0, value=1.0, step=.01, key=f"SOL {dep}{ind}{j}") 
                                                    for j in range(order))
                    eq = EulerCauchy(coefficients)
                    eq.constants = constants
                    eqs[f'{dep}{ind}'] = (eq,)
                    st.latex(display_solution(eq, constants))

with image:
    generated_image = generate_image(*tuple(eqs.values()))
    st.image(generated_image, use_column_width=True)