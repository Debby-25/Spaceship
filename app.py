import streamlit as st
import pandas as pd
import joblib

# Load the reduced model and top features
model = joblib.load('spaceship_model_reduced.pkl')
reference_features = joblib.load('spaceship_columns_reduced.pkl')

st.set_page_config(page_title="Spaceship Titanic Predictor", layout="centered")
st.title("Spaceship Titanic Prediction")
st.markdown("Enter a passenger’s details to predict whether they were transported to another dimension.")

# Sidebar info
st.sidebar.title("SPACESHIP TITANIC PREDICTION")
st.sidebar.info("""USER'S TO PREDICT EITHER PASSENGER WERE TRANSPORTED OR NOT""")

# Input form
with st.form("manual_entry_form"):
    st.subheader("Passenger Details")

    col1, col2 = st.columns(2)
    with col1:
        homeplanet = st.selectbox("Home Planet", ['Earth', 'Europa', 'Mars'])
        destination = st.selectbox("Destination", ['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'])
        age = st.slider("Age", 0, 100, 30)
        cryosleep = st.radio("In CryoSleep?", ['True', 'False'])
        vip = st.radio("VIP Status", ['True', 'False'])

    with col2:
        deck = st.selectbox("Cabin Deck", ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])
        side = st.selectbox("Cabin Side", ['P', 'S'])
        room_service = st.number_input("Room Service", min_value=0, step=10)
        food_court = st.number_input("Food Court", min_value=0, step=10)
        shopping_mall = st.number_input("Shopping Mall", min_value=0, step=10)
        spa = st.number_input("Spa", min_value=0, step=10)
        vrdeck = st.number_input("VR Deck", min_value=0, step=10)

    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    input_dict = {
        'HomePlanet': homeplanet,
        'Destination': destination,
        'Age': age,
        'CryoSleep': cryosleep == 'True',
        'VIP': vip == 'True',
        'Deck': deck,
        'Side': side,
        'RoomService': room_service,
        'FoodCourt': food_court,
        'ShoppingMall': shopping_mall,
        'Spa': spa,
        'VRDeck': vrdeck
    }

    input_df = pd.DataFrame([input_dict])
    st.subheader("Passenger Preview")
    st.write(input_df)

    # Encode and align
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=reference_features, fill_value=0)

    pred = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    if pred:
        st.success("✅ This passenger **was transported** to another dimension!")
    else:
        st.warning("❌ This passenger **was not transported.**")