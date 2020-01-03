# Import Packages
import numpy as np
import pandas as pd 
import joblib 
import streamlit as st

#data_load_state = st.text('Loading data...')

#data_load_state.text('Loading data...done!')



# Title of your Web App
st.title('Sales Forecasting')

# Describe your Web App
st.markdown("We demonstrate how we can forecast advertising sales based on ads expenditure.")


# Read Data
data = pd.read_csv('data/advertising_regression.csv')

#cols = ["TV", "radio", "newspaper", "sales"]
#st_ms = st.multiselect("Columns", data.columns.tolist(), default=cols)

# Show Data
##data

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
   

pics = {
    "TV": "https://s3.eu-central-1.amazonaws.com/centaur-wp/econsultancy/prod/content/uploads/archive/images/resized/0001/0980/tv_ads_11-blog-half.png",
    "radio": "https://www.radiotoday.com.au/wp-content/uploads/CRA/number-crunching-jan-2017.jpg",
    "newspaper": "https://mediaspectrum.net/wp-content/uploads/2015/03/advantages-of-newspaper-advertising-1080x675.jpg"
}
pic = st.selectbox("Advertising Channel", list(pics.keys()), 0)
st.image(pics[pic], use_column_width=True, caption=pics[pic])


# Add sliders and assign them to variables
st.sidebar.subheader('Advertising Costs')

# TV Slider
TV = st.sidebar.slider('TV Advertising Cost', 0, 300, 150) # (Title, min value, max value, default value)

# Radio Slider
radio = st.sidebar.slider('Radio Advertising Cost', 0, 50, 25) # (Title, min value, max value, default value)

# Newspaper Slider
newspaper = st.sidebar.slider('Newspaper Advertising Cost', 0, 250, 125) # (Title, min value, max value, default value)


# Let's Draw A Histogram

### TV
st.subheader('TV Ad Cost Distribution')

# Use numpy to generate bins for age
hist_values = np.histogram(data.TV, bins=300, range=(0,300))[0]

# Show Bar Chart
st.bar_chart(hist_values)

### Newspaper
st.subheader('Newspaper Ad Cost Distribution')

# Use numpy to generate bins for age
hist_values = np.histogram(data.newspaper, bins=300, range=(0,300))[0]

# Show Bar Chart
st.bar_chart(hist_values)

### Radio
st.subheader('Radio Ad Cost Distribution')

# Use numpy to generate bins for age
hist_values = np.histogram(data.radio, bins=300, range=(0,300))[0]

# Show Bar Chart
st.bar_chart(hist_values)

### Sales
st.subheader('Historical Sales Distribution')

# Use numpy to generate bins for age
hist_values = np.histogram(data.sales, bins=300, range=(0,300))[0]

# Show Bar Chart
st.bar_chart(hist_values)



# Load saved machine learning model
st.subheader("Predicted Sales")

# Load model using joblib
saved_model = joblib.load('advertising_model.sav')

# Predict sales using variables/features
predicted_sales = saved_model.predict([[TV, radio, newspaper]])[0]

# Print prediction
st.write(f"Predicted sales is {int(predicted_sales*1000)} dollars.")