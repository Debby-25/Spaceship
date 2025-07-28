import streamlit as st
import pandas as pd
import joblib

# Load the reduced model and reference features
model = joblib.load('spaceship_model_reduced.pkl')
reference_features = joblib.load('spaceship_columns_reduced.pkl')

st.set_page_config(page_title="Spaceship Titanic Predictor", layout="centered")
st.title("Spaceship Titanic Prediction")
st.markdown("Enter passenger details or upload a CSV to predict who was transported.")

# Sidebar info
st.sidebar.title("SPACESHIP TITANIC")
st.sidebar.info("USER'S TO PREDICT EITHER PASSENGER WERE TRANSPORTED OR NOT.")

# --- CSV Upload ---
st.sidebar.subheader("📂 Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    st.subheader("📊 Batch Prediction Result")
    batch_df = pd.read_csv(uploaded_file)
    batch_encoded = pd.get_dummies(batch_df)
    batch_encoded = batch_encoded.reindex(columns=reference_features, fill_value=0)
    batch_preds = model.predict(batch_encoded)
    batch_df['Transported'] = batch_preds.astype(bool)
    st.write(batch_df)
    st.download_button("Download Prediction CSV", batch_df.to_csv(index=False), file_name="batch_prediction.csv")

# --- Manual Entry Form ---
with st.form("manual_entry_form"):
    st.subheader("🧍 Passenger Details")

    col1, col2 = st.columns(2)
    with col1:
        homeplanet = st.selectbox("Home Planet", ['Earth', 'Europa', 'Mars'])
        destination = st.selectbox("Destination", ['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'])
        age = st.slider("Age", 0, 100, 30)
        cryosleep = st.radio("CryoSleep", ['True', 'False'])
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

# --- Prediction Logic ---
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

    encoded_df = pd.get_dummies(input_df)
    encoded_df = encoded_df.reindex(columns=reference_features, fill_value=0)

    pred = model.predict(encoded_df)[0]
    pred_proba = model.predict_proba(encoded_df)[0]
    confidence = round(pred_proba[int(pred)] * 100, 2)

    st.subheader("Prediction Result")
    if pred:
        st.success("✅ This passenger **was transported** to another dimension!")
    else:
        st.warning("❌ This passenger **was not transported.**")

    st.metric("Confidence Level", f"{confidence}%")