import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Welcome to Data Analysis")
st.badge(
	"This Application is designed to help predict house price using sample data", 
	color="blue")

uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

if uploaded_file is not None:
	df =  pd.read_csv(uploaded_file)
	st.subheader('1. Preview of the uploaded file')
	st.dataframe(df.head(5))

	st.subheader('2. Summary Statistic of the File')
	st.write(df.describe())

	st.subheader('3. Show all Columns')
	st.write(df.columns)

	st.subheader('4. Attrition Count')
	attrition_count = df['Attrition'].value_counts()
	st.write(attrition_count)
	st.bar_chart(attrition_count)

	# Filter By Department
	st.subheader("5. Filter by department")
	if 'Department' in df.columns:
		dept = st.selectbox("Select a Department", df['Department'].unique())
		st.dataframe(df[df['Department']== dept])

	# Correlation Matrix
	st.subheader('6. Correlation Heatmap')
	numerical_col = df.select_dtypes(include=['int64','float64'])
	corr =  numerical_col.corr()
	st.write(corr)

	fig, ax = plt.subplots(figsize=(12, 8))
	sns.heatmap(corr, annot=True, fmt=".1f", cmap="coolwarm", ax=ax)
	st.pyplot(fig)

	














with st.echo():
	st.write('This project helps predict house price using linear regression')