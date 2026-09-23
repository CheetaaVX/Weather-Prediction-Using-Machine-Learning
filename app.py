import streamlit as st
import numpy as np
import joblib
import requests


model = joblib.load("weather_summary_model.pkl")
le = joblib.load("label_encoder.pkl")


st.set_page_config(
    page_title="Weather Prediction ML App",
    page_icon="^_^",
    layout="centered"
)

st.markdown("""
<h1 style='text-align: center;'> Weather Prediction using Machine Learning</h1>
<hr>
""", unsafe_allow_html=True)

# add weather api key here
API_KEY = ""

st.subheader(" Enter City for Live Weather")
city_input = st.text_input("City Name", "$cityname")  # default City name

def get_weather(city_name):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city_name}"
    try:
        res = requests.get(url)
        data = res.json()
        current = data["current"]

        return {
            "temp": current["temp_c"],
            "humidity": current["humidity"] / 100,
            "wind": current["wind_kph"],
            "pressure": current["pressure_mb"],
            "visibility": current["vis_km"],
            "cloud": current["cloud"],
            "precip": current["precip_mm"],
            "condition": current["condition"]["text"]
        }
    except:
        return None

# 
if st.button(" Use Live Weather Data"):
    weather_data = get_weather(city_input)

    if weather_data:
        st.session_state.temp = weather_data["temp"]
        st.session_state.humidity = weather_data["humidity"]
        st.session_state.wind = weather_data["wind"]
        st.session_state.pressure = weather_data["pressure"]
        st.session_state.visibility = weather_data["visibility"]
        st.session_state.cloud = weather_data["cloud"]

        if weather_data["precip"] > 0:
            st.session_state.precip = "Rain"
        else:
            st.session_state.precip = "None"

        st.success(f" Live data loaded for {city_input}: {weather_data['condition']}")
    else:
        st.error(" Failed to fetch weather data. Check city name!")


st.subheader(" Enter / Modify Weather Parameters")

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input(
        " Temperature (°C)", -30.0, 50.0,
        value=st.session_state.get("temp", 25.0)
    )

    humidity = st.slider(
        " Humidity", 0.0, 1.0,
        value=st.session_state.get("humidity", 0.5)
    )

    wind_speed = st.number_input(
        " Wind Speed (km/h)", 0.0, 150.0,
        value=st.session_state.get("wind", 10.0)
    )

    hour = st.slider(" Hour of Day", 0, 23, 12)

with col2:
    pressure = st.number_input(
        " Pressure (millibars)", 800.0, 1100.0,
        value=st.session_state.get("pressure", 1013.0)
    )

    visibility = st.number_input(
        " Visibility (km)", 0.0, 20.0,
        value=st.session_state.get("visibility", 10.0)
    )

    precip_type = st.selectbox(
        " Precipitation Type",
        ["None", "Rain", "Snow"],
        index=["None", "Rain", "Snow"].index(
            st.session_state.get("precip", "None")
        )
    )


st.markdown("###  Sky Condition Control")

cloud_cover = st.slider(
    " Cloud Cover (%)", 0, 100,
    value=st.session_state.get("cloud", 0)
)


precip_map = {"None": 0, "Rain": 1, "Snow": 2}
precip_enc = precip_map[precip_type]

temp_humidity = temperature * humidity

input_data = np.array([[temperature, humidity, wind_speed,
                        pressure, visibility, hour,
                        precip_enc, temp_humidity]])


st.markdown("<br>", unsafe_allow_html=True)

if st.button(" Predict Weather", use_container_width=True):

    prediction = model.predict(input_data)
    weather = le.inverse_transform(prediction)[0]

    # Confidence
    try:
        proba = model.predict_proba(input_data)
        confidence = np.max(proba) * 100
    except:
        confidence = None

    # Smart cloud adjustment 
    if weather not in ["Rain", "Snow"]:
        if cloud_cover < 30:
            weather = "Clear"
        elif cloud_cover > 60:
            weather = "Cloudy"

    # day/night logic
    if weather == "Clear" and 6 <= hour <= 18:
        weather = "Sunny"

    # o/p
    st.markdown(f"""
        <div style="
            background-color:#f0f8ff;
            padding:25px;
            border-radius:15px;
            text-align:center;
            font-size:26px;
            font-weight:bold;
            color:#003366;
        ">
         Predicted Weather Condition <br><br>
        <span style='font-size:32px;'>{weather}</span>
        </div>
    """, unsafe_allow_html=True)

    if confidence is not None:
        st.markdown(f"""
        <p style='text-align:center; font-size:18px;'>
         Model Confidence: <b>{confidence:.2f}%</b>
        </p>
        """, unsafe_allow_html=True)

#foot
st.markdown("""
<br><hr>
<p style='text-align:center; color:gray;'>

</p>
""", unsafe_allow_html=True)