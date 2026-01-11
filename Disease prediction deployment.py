import pandas as pd
import numpy as np
import joblib
import streamlit as st


parkinson_model = joblib.load(r"OneDrive/Desktop/GUVI/Projects/Models/Parkinson/XGBC_Parkinson.pkl")
parkinson_scaler = joblib.load(r"OneDrive/Desktop/GUVI/Projects/Models/Parkinson/XGBC_Parkinson_scaler.pkl")
parkinson_features = joblib.load(r"OneDrive/Desktop/GUVI/Projects/Models/Parkinson/XGBC_feature_names_Parkinson.pkl")

Kidney_model = joblib.load(r"OneDrive/Desktop/GUVI/Projects/Models/Kidney/XGBC_Kidney.pkl")
Kidney_encoder = joblib.load(r"OneDrive/Desktop/GUVI/Projects/Models/Kidney/XGBC_kidney_encoder.pkl")
Kidney_Scaler = joblib.load(r"OneDrive/Desktop/GUVI/Projects/Models/Kidney/XGBC_kidney_scaler.pkl")
Kidney_columns = joblib.load(r"OneDrive/Desktop/GUVI/Projects/Models/Kidney/XGBC_feature_names_kidney.pkl")

Liver_model = joblib.load(r"OneDrive/Desktop/GUVI/Projects/Models/Liver/LR_Liver.pkl")
Liver_encoder = joblib.load(r"OneDrive/Desktop/GUVI/Projects/Models/Liver/LR_Liver_encoder.pkl")
Liver_scaler = joblib.load(r"OneDrive/Desktop/GUVI/Projects/Models/Liver/LR_Liver_scaler.pkl")
Liver_columns = joblib.load(r"OneDrive/Desktop/GUVI/Projects/Models/Liver/LR_feature_names_Liver.pkl")


s = st.sidebar.radio(
    "⚕️Multi Disease Prediction",
    (
        "🏡Home",
        "🧠Parkinsons Disease Prediction",
        "🩺💧 Kidney Disease Prediction",
        "⚕️Liver Disease Prediction",
    ),
)


if s == "🏡Home":
    st.title("⚕️Multi Disease Prediction")
    
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    
    st.image(r"C:\Users\Dell\OneDrive\Desktop\GUVI - PROJECTS\Project 4_Disease Prediction\human-anatomy-organs-brain-kidney.jpg")


elif s == "🧠Parkinsons Disease Prediction":
    st.title("🧠Parkinsons Disease Prediction")

    MDVP_Fo_Hz = st.number_input("MDVP:Fo(Hz)", min_value=50.000, max_value=900.000, step=0.001, format="%.3f")
    MDVP_Fhi_Hz = st.number_input("MDVP:Fhi(Hz)", min_value=100.000, max_value=999.000, step=0.001, format="%.3f")
    MDVP_Flo_Hz = st.number_input("MDVP:FLo(Hz)", min_value=10.000, max_value=999.000, step=0.001, format="%.3f")
    MDVP_Jitter = st.number_input("MDVP:Jitter(%)", min_value=0.00000, max_value=0.10000, step=0.00001, format="%.5f")
    MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.00000, max_value=0.10000, step=0.00001, format="%.5f")
    MDVP_RAP = st.number_input("MDVP:RAP", min_value=0.00000, max_value=0.10000, step=0.00001, format="%.5f")
    MDVP_PPQ = st.number_input("MDVP_PPQ", min_value=0.00000, max_value=0.10000, step=0.00001, format="%.5f")
    Jitter_DDP = st.number_input("Jitter:DDP", min_value=0.00000, max_value=0.10000, step=0.00001, format="%.5f")
    MDVP_Shimmer = st.number_input("MDVP_Shimmer", min_value=0.00000, max_value=0.10000, step=0.00001, format="%.5f")
    MDVP_Shimmer_dp = st.number_input("MDVP_Shimmer_dp", min_value=0.000, max_value=0.999, step=0.001, format="%.3f")
    Shimmer_APQ3 = st.number_input("Shimmer:APQ3", min_value=0.00000, max_value=0.10000, step=0.00001, format="%.5f")
    Shimmer_APQ5 = st.number_input("Shimmer:APQ5", min_value=0.00000, max_value=0.10000, step=0.00001, format="%.5f")
    MDVP_APQ = st.number_input("MDVP_APQ", min_value=0.00000, max_value=0.10000, step=0.00001, format="%.5f")
    Shimmer_DDA = st.number_input("Shimmer:DDA", min_value=0.00000, max_value=0.10000, step=0.00001, format="%.5f")
    NHR = st.number_input("NHR", min_value=0.00000, max_value=0.90000, step=0.00001, format="%.5f")
    HNR = st.number_input("HNR", min_value=1.000, max_value=99.000, step=0.001, format="%.3f")
    RPDE = st.number_input("RPDE", min_value=0.100000, max_value=0.900000, step=0.00001, format="%.6f")
    DFA = st.number_input("DFA", min_value=0.100000, max_value=0.900000, step=0.00001, format="%.6f")
    spread1 = st.number_input("spread1", min_value=-9.00000, max_value=-1.00000, step=0.00001, format="%.5f")
    spread2 = st.number_input("spread2", min_value=0.000000, max_value=100.000000, step=0.000001, format="%.6f")
    D2 = st.number_input("D2", min_value=0.00000, max_value=9.000000, step=0.00001, format="%.6f")
    PPE = st.number_input("PPE", min_value=0.000000, max_value=0.900000, step=0.000001, format="%.6f")

    input_df = pd.DataFrame(
        [[
            MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz,
            MDVP_Jitter, MDVP_Jitter_Abs, MDVP_RAP,
            MDVP_PPQ, Jitter_DDP, MDVP_Shimmer,
            MDVP_Shimmer_dp, Shimmer_APQ3,
            Shimmer_APQ5, MDVP_APQ, Shimmer_DDA,
            NHR, HNR, RPDE, DFA,
            spread1, spread2, D2, PPE
        ]],
        columns=parkinson_features
    )

    input_scaled = parkinson_scaler.transform(input_df)

    if st.button("🧠 Parkinsons Disease Prediction"):
        pred = parkinson_model.predict(input_scaled)[0]
        prob = parkinson_model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.error(f"⚠️ Parkinsons detected (probability: {prob:.2f})")
        else:
            st.success(f"✅ Healthy (probability of Parkinsons: {prob:.2f})")


elif s == "🩺💧 Kidney Disease Prediction":
    st.title("🩺💧 Kidney Disease Prediction")

    Age = st.number_input("Age", min_value=1, max_value=100, step=1)
    bp = st.number_input("Blood pressure", min_value=0, max_value=250, step=1)
    sg = st.number_input("Specific Gravity", min_value=0.1, max_value=2.0, step=0.1, format="%.2f")
    al = st.number_input("Album", min_value=0, max_value=10, step=1)
    su = st.number_input("Sugar", min_value=0, max_value=10, step=1)

    pc = st.selectbox("Pus Cell", Kidney_encoder["pc"].classes_)
    bgr = st.number_input("Blood Glucose", min_value=1, max_value=100, step=1)
    bu = st.number_input("Blood Urea", min_value=0.0, max_value=1000.0, step=0.1, format="%.2f")
    sc = st.number_input("Serum Creatinine", min_value=0.0, max_value=1000.0, step=0.1, format="%.2f")
    sod = st.number_input("Sodium", min_value=0.1, max_value=1000.0, step=0.1, format="%.3f")
    pot = st.number_input("Potassium", min_value=0.1, max_value=100.0, step=0.1, format="%.2f")
    Hemo = st.number_input("Hemoglobin", min_value=0.1, max_value=20.0, step=0.1, format="%.1f")
    pcv = st.number_input("Packed Cell Volume", min_value=1, max_value=100, step=1)
    wc = st.number_input("White Blood Cell Count", min_value=1000, max_value=50000, step=1)
    rc = st.number_input("Red Blood Cell Count", min_value=0.1, max_value=100.0, step=0.1, format="%.2f")

    htn = st.selectbox("Hypertension", Kidney_encoder["htn"].classes_)
    dm = st.selectbox("Diabetes", Kidney_encoder["dm"].classes_)
    appet = st.selectbox("Appetite", Kidney_encoder["appet"].classes_)
    pe = st.selectbox("Pedal Edema", Kidney_encoder["pe"].classes_)
    ane = st.selectbox("Anemia", Kidney_encoder["ane"].classes_)

    pc = Kidney_encoder["pc"].transform([pc])[0]
    htn = Kidney_encoder["htn"].transform([htn])[0]
    dm = Kidney_encoder["dm"].transform([dm])[0]
    appet = Kidney_encoder["appet"].transform([appet])[0]
    pe = Kidney_encoder["pe"].transform([pe])[0]
    ane = Kidney_encoder["ane"].transform([ane])[0]

    Kinput_df = pd.DataFrame(
        [[
            Age, bp, sg, al, su,
            Kidney_encoder["rbc"].transform(["normal"])[0],
            pc,
            Kidney_encoder["pcc"].transform(["notpresent"])[0],
            Kidney_encoder["ba"].transform(["notpresent"])[0],
            bgr, bu, sod, pot, Hemo, pcv, wc, sc, rc,
            htn, dm,
            Kidney_encoder["cad"].transform(["no"])[0],
            appet, pe, ane
        ]],
        columns=Kidney_columns
    )

    kinput_scaled = Kidney_Scaler.transform(Kinput_df)

    if st.button("🩺💧 Kidney Disease Prediction"):
        pred = Kidney_model.predict(kinput_scaled)[0]
        prob = Kidney_model.predict_proba(kinput_scaled)[0][1]

        if pred == 1:
            st.error(f"⚠️ Kidney disease detected (probability: {prob:.2f})")
        else:
            st.success(f"✅ Healthy (probability of Kidney Disease: {prob:.2f})")

elif s == "⚕️Liver Disease Prediction":
    st.title("⚕️Liver Disease Prediction")
    
    age = st.number_input("Age", 1, 100)
    Gender = st.selectbox("Gender", Liver_encoder["Gender"].classes_)
    Total_Bilirubin = st.number_input("Total_Bilirubin", 0.1, 100.0)
    Direct_Bilirubin = st.number_input("Direct_Bilirubin", 0.1, 20.0)
    Alkaline_Phosphotase = st.number_input("Alkaline_Phosphotase", 0, 5000)
    Alamine_Aminotransferase = st.number_input("Alamine_Aminotransferase", 0, 3000)
    Aspartate_Aminotransferase = st.number_input("Aspartate_Aminotransferase", 1, 5000)
    Total_Protiens = st.number_input("Total_Protiens", 0.1, 10.0, step = 0.1, format = "%.2f")
    Albumin = st.number_input("Albumin", 0.1, 10.0, step = 0.1)
    Albumin_and_Globulin_Ratio = st.number_input("Albumin_and_Globulin_Ratio", 0.1, 5.0, step = 0.1)
    
    Gender = Liver_encoder["Gender"].transform([Gender])[0]
    
    linput_df = pd.DataFrame([[
        age, Gender, Total_Bilirubin,
        Direct_Bilirubin,
        Alkaline_Phosphotase,
        Alamine_Aminotransferase,
        Aspartate_Aminotransferase,
        Total_Protiens,
        Albumin,
        Albumin_and_Globulin_Ratio
        ]],columns = Liver_columns)
    
    linput_scaled = Liver_scaler.transform(linput_df)
    
    if st.button("⚕️Liver Disease Prediction"):
        pred = Liver_model.predict(linput_scaled)[0]
        prob = Liver_model.predict_proba(linput_scaled)[0][1]

        if pred == 1:
            st.error(f"⚠️ Liver disease detected (probability: {prob:.2f})")
        else:
            st.success(f"✅ Healthy (probability of Liver Disease: {prob:.2f})")
    
        
    
    
