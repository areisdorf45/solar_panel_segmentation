import streamlit as st
import pandas as pd

st.write("Hello !!!!!")

st.write("Here's our first attempt at using data to create a table:")

st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))
xx = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x) = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)
import streamlit as st
import numpy as np
import pandas as pd

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))

st.dataframe(dataframe.style.highlight_max(axis=0))
