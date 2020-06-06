import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

def analytical_solution(H, cv, uo, T_range, drainage):

  T = np.linspace(T_range[0], T_range[1], 10)
  N = 50
  u = np.zeros((len(T), N))

  if drainage == 'Top and bottom':
    h = H/2.0
    z = np.linspace(-h, h, N)
    u[:,0] = 0.0
    u[:,-1] = 0.0
    z_start = 1
    z_end = len(z) - 1
  elif drainage == 'Top only':
    h = H
    z = np.linspace(0, h, N)
    u[:,-1] = 0.0
    z_start = 0
    z_end = len(z) - 1
  else:
    h = H
    z = np.linspace(-h, 0, N)
    u[:,0] = 0.0
    z_start = 1
    z_end = len(z)

  u[:,z_start:z_end] = uo

  term1 = np.pi * z[z_start:z_end] / (2*h)
  for i in range(len(T)):
    term2 = (np.pi/2)**2 * T[i]
    sum1 = 0
    for k in range(1, 100):
      factor = ((-1.0)**(k-1) / (2*k-1))
      sum1 += factor * np.cos((2*k-1) * term1) * np.exp(-1.0 * (2*k-1)**2 * term2)
    u[i,z_start:z_end] = uo * 4.0/np.pi * sum1

  Tu = np.linspace(T_range[0], T_range[1], 1000)
  t = np.array([i*h**2/(cv * 0.00273973) for i in Tu])

  sum2 = np.zeros(len(Tu))
  for k in range(1, 100):
    factor2 = 1.0 / (2*k - 1)**2
    sum2 += factor2 * np.exp(-1.0 * (2*k-1)**2 * (np.pi/2)**2 * Tu)
  sum2 = 1 - (8/np.pi**2) * sum2

  U = sum2 * 100
  y = np.linspace(H, 0, N)

  yu = np.hstack((y[:,None], u.T))
  tU = np.hstack((t[:,None], U[:,None]))

  return yu, tU

st.title('Consolidation 1D: Interactive Demo')

bc = st.sidebar.selectbox(
  label="Drained boundaries",
  options=(
    'Top only',
    'Bottom only',
    'Top and bottom'
  )
)

H = st.sidebar.number_input(
  'Layer thickness (m)',
  min_value=1,
  step=1,
  value=2
)

u0 = st.sidebar.number_input(
  'Initial excess pore pressure (kPa)',
  min_value=50,
  step=25,
  value=100
)

cv = st.sidebar.number_input(
  'Coefficient of consolidation (m2/yr)',
  min_value=0.01,
  max_value=10.0,
  value=0.1,
  step=0.1
)

T = st.sidebar.slider("Dimensionless time range, T", 0.01, 2.0, (0.01, 1.0))

yu, tU = analytical_solution(H, cv, u0, T, bc)

cols1 = ['H (m)'] + [str(round(i, 2)) for i in np.linspace(T[0], T[1], 10)]
df1 = pd.DataFrame(yu, columns=cols1)
df1_melt = df1.melt(
  id_vars='H (m)',
  value_vars=cols1[1:],
  var_name='Dimensionless time, T',
  value_name='Pore pressure, u (kPa)'
)

fig1 = px.line(df1_melt,
  x='Pore pressure, u (kPa)',
  y='H (m)',
  color='Dimensionless time, T'
)
fig1.layout.update(
  title='Excess Pore Pressure Dissipation',
  title_x=0.0,
  showlegend=False
)
fig1.update_yaxes(autorange="reversed")
st.write(fig1)

cols2 = ['Real time, t (days)', 'Degree of Consolidation, U (%)']
df2 = pd.DataFrame(tU, columns=cols2)

fig2 = px.line(df2,
  x='Real time, t (days)',
  y='Degree of Consolidation, U (%)',
  log_x=True
)
fig2.update_yaxes(autorange="reversed")
fig2.layout.update(
  title='Progress of Consolidation',
  title_x=0.0
)
st.write(fig2)

if st.sidebar.checkbox("Show governing equation", value=True):
  st.markdown("""
  ## Governing Equation
  The partial differential equation (PDE) governing one-dimensional fluid flow in a porous medium is given by
  $$
  \\frac{\partial u}{\partial t} - c_v \\frac{\partial^2 u}{\partial z^2} = 0
  $$
  where $u$ is the pore fluid pressure, $t$  stands for time, $z$ represents depth and $c_v$ is the coefficient of consolidation which can be expressed as
  $$
  c_v = \\frac{k}{m_v \gamma}
  $$
  with $k$ being the hydraulic conductivity, $m_v$ the coefficient of volumetric compressibility and $\gamma$ the unit weight of the fluid. From known elastic parameters of the porous medium, $m_v$ can be calculated as
  $$
  m_v = \\frac{(1+\\nu) (1-2\\nu)}{E (1-\\nu)}
  $$
  where $E$ is the Young's modulus and $\\nu$ is the Poisson's ratio of the porous medium. Thus, the coefficient of consolidation can be calculated for a known hydraulic conductivity and elastic parameters.
  """)

if st.sidebar.checkbox("Show analytical solution", value=True):
  st.markdown("""
  ## Analytical Solution
  The analytical solution of the PDE is given by
  $$
  \\frac{u(t,z)}{u_o} = \\frac{4}{\pi} \sum_{i=1}^\infty
  \\frac{{(-1)}^{i-1}}{2i-1} \exp{\left[ -{(2i-1)}^2 \\frac{\pi^2 T}{4} \\right]}
  \cos{\left[ (2i-1) \\frac{\pi z}{2h} \\right]}
  $$
  where $u_o$ is the initial excess pore pressure, $h$ is the drainage height and $T$ is the dimensionless time defined in terms of the coefficient of consolidation and the real time as
  $$
  T = \\frac{c_v t}{h^2}
  $$
  If both boundaries are drained, $h=0.5H$ and if only one boundary is drained, $h=H$, where $H$ is the height of the model.
  The degree of consolidation $U$, expressed in percentage, indicates the level of consolidation achieved at a given time with respect to what is achievable for a given stress condition. The analytical solution for $U$ is given by
  $$
  U = 1 - \\frac{8}{\pi^2} \sum_{i=1}^\infty \\frac{1}{(2i-1)^2}  \exp{\left[ -{(2i-1)}^2 \\frac{\pi^2 T}{4} \\right]}
  $$
  """)

st.sidebar.markdown(  
  '''
  [GitHub](https://github.com/yaredwb)
  [LinkedIn](https://www.linkedin.com/in/yaredworku/) 
  '''
)

