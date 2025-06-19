import streamlit as st, pickle, numpy as np, pandas as pd, json, shap

# Loading the model
model = pickle.load(open("model_xgb.sv",'rb'))
rent_model = pickle.load(open("rent_model_xgb.sv",'rb'))
with open("category_mappings.json") as f:
    category_mappings = json.load(f)

st.set_page_config(page_title="Cena mieszkań w Polsce")
st.title("Cena mieszkań w Polsce")
t_predictor, t_input, t_info = st.tabs(["Predykcja ceny mieszkań", "Trenowanie modelu", "Informacje o projekcie"])

type_d = {
    "apartmentBuilding": "Apartamentowiec",
    "blockOfFlats": "Blok mieszkalny",
    "tenement": "Kamienica",
}
buildingMaterial_d = {
    "brick": "Cegła",
    "concreteSlab": "Beton",
}
condition_d = {
    "low": "Do remontu",
    "premium": "Nowy",
}
ownership_d = {
    "condominium": "Własność kondominium",
    "cooperative": "Spółdzielnia mieszkaniowa",
    "udział": "Udział", # Just why
}
city_d = {
    "bialystok": "Białystok",
    "bydgoszcz": "Bydgoszcz",
    "czestochowa": "Częstochowa",
    "gdansk": "Gdańsk",
    "gdynia": "Gdynia",
    "katowice": "Katowice",
    "krakow": "Kraków",
    "lodz": "Łódź",
    "lublin": "Lublin",
    "poznan": "Poznań",
    "radom": "Radom",
    "rzeszow": "Rzeszów",
    "szczecin": "Szczecin",
    "warsaw": "Warszawa",
    "wroclaw": "Wrocław",
}
def yesify(x):
    return "yes" if x else "no"
def rentify(x):
    return f"{'rent_' if isRent else ''}{x}"

def retrain(data, inc):
    df = pd.read_csv(data)
    df = df.drop(columns=["id","latitude","longitude"])
    for column in ["city", "type", "ownership", "buildingMaterial", "condition", "hasParkingSpace", "hasBalcony", "hasElevator", "hasSecurity", "hasStorageRoom"]:
        df[column] = df[column].astype("category")
    if isRent:
        rent_model.fit(df.drop(columns=["price"]), df["price"], xgb_model=rent_model.get_booster() if inc else None)
        pickle.dump(rent_model, open("rent_model_xgb.sv",'wb'))
    else:
        model.fit(df.drop(columns=["price"]), df["price"], xgb_model=model.get_booster() if inc else None)
        pickle.dump(model, open("model_xgb.sv",'wb'))
    st.success("Model został wytrenowany na podstawie danych z pliku.")

def untrain():
    if isRent:
        rent_model = pickle.load(open("rent_model_xgb_original.sv",'rb'))
        pickle.dump(rent_model, open("rent_model_xgb.sv",'wb'))
    else:
        model = pickle.load(open("model_xgb_original.sv",'rb'))
        pickle.dump(model, open("model_xgb.sv",'wb'))
    st.success("Model został zresetowany do stanu oryginalnego.")

with t_predictor:
    city = st.selectbox("Miasto", city_d.keys(), format_func=lambda x: city_d[x])
    squareMeters = st.number_input("Powierzchnia (m²)", min_value=1, max_value=1000, value=50)

    l, m, r = st.columns(3)
    with l:
        type_ = st.selectbox("Typ budynku", [None, *type_d.keys()], format_func=lambda x: "--- Wybierz ---" if x is None else type_d[x])
        buildingMaterial = st.selectbox("Materiał", [None, *buildingMaterial_d.keys()], format_func=lambda x: "--- Wybierz ---" if x is None else buildingMaterial_d[x])
        condition = st.selectbox("Kondycja", [None, *condition_d.keys()], format_func=lambda x: "--- Wybierz ---" if x is None else condition_d[x])
    with m:
        floorCount = st.number_input("Liczba pięter w budynku", min_value=1, max_value=100, value=None)
        rooms = st.number_input("Liczba pokoi", min_value=1, max_value=10, value=None)
        floor = st.number_input("Piętro mieszkania", min_value=0, max_value=100, value=None)
    with r:
        ownership = st.selectbox("Właściciel", [None, *ownership_d.keys()], format_func=lambda x: "--- Wybierz ---" if x is None else ownership_d[x])
        buildYear = st.number_input("Rok budowy", min_value=1900, max_value=2024, value=None)
        poiCount = st.number_input("Liczba punktów usługowych w okolicy", min_value=0, max_value=100, value=None)

    with st.expander("Dodatkowe opcje"):
        c_distance, c_addons = st.columns([2,1])
        with c_distance:
            st.write("Odległości do (km):")
            left, right = st.columns(2)
            with left:
                centreDistance = st.number_input("Centrum miasta", min_value=0.0, max_value=100.0, value=None)
                schoolDistance = st.number_input("Szkoła", min_value=0.0, max_value=100.0, value=None)
                clinicDistance = st.number_input("Przychodnia", min_value=0.0, max_value=100.0, value=None)
                postOfficeDistance = st.number_input("Poczta", min_value=0.0, max_value=100.0, value=None)
            with right:
                kindergartenDistance = st.number_input("Przedszkole", min_value=0.0, max_value=100.0, value=None)
                restaurantDistance = st.number_input("Restauracja", min_value=0.0, max_value=100.0, value=None)
                collegeDistance = st.number_input("Uczelnia", min_value=0.0, max_value=100.0, value=None)
                pharmacyDistance = st.number_input("Apteka", min_value=0.0, max_value=100.0, value=None)
        with c_addons:
            st.write("Dodatki:")
            hasParkingSpace = st.checkbox("Parking")
            hasBalcony = st.checkbox("Balkon")
            hasElevator = st.checkbox("Winda")
            hasSecurity = st.checkbox("Ochrona")
            hasStorageRoom = st.checkbox("Pomieszczenie gospodarcze")
    
    input_df = pd.DataFrame([{
        "city": city,
        "type": type_,
        "squareMeters": squareMeters,
        "rooms": rooms,
        "floor": floor,
        "floorCount": floorCount,
        "buildYear": buildYear,
        "centreDistance": centreDistance,
        "poiCount": poiCount,
        "schoolDistance": schoolDistance,
        "clinicDistance": clinicDistance,
        "postOfficeDistance": postOfficeDistance,
        "kindergartenDistance": kindergartenDistance,
        "restaurantDistance": restaurantDistance,
        "collegeDistance": collegeDistance,
        "pharmacyDistance": pharmacyDistance,
        "ownership": ownership,
        "buildingMaterial": buildingMaterial,
        "condition": condition,
        "hasParkingSpace": yesify(hasParkingSpace),
        "hasBalcony": yesify(hasBalcony),
        "hasElevator": yesify(hasElevator),
        "hasSecurity": yesify(hasSecurity),
        "hasStorageRoom": yesify(hasStorageRoom)
    }]).astype({
        "city": "category",
        "type": "category",
        "squareMeters": float,
        "rooms": float,
        "floor": float,
        "floorCount": float,
        "buildYear": float,
        "centreDistance": float,
        "poiCount": float,
        "schoolDistance": float,
        "clinicDistance": float,
        "postOfficeDistance": float,
        "kindergartenDistance": float,
        "restaurantDistance": float,
        "collegeDistance": float,
        "pharmacyDistance": float,
        "ownership": "category",
        "buildingMaterial": "category",
        "condition": "category",
        "hasParkingSpace": "category",
        "hasBalcony": "category",
        "hasElevator": "category",
        "hasSecurity": "category",
        "hasStorageRoom": "category"
    })

    for col in ["city", "type", "ownership", "buildingMaterial", "condition", "hasParkingSpace", "hasBalcony", "hasElevator", "hasSecurity", "hasStorageRoom"]:
        input_df[col] = pd.Categorical(input_df[col], categories=category_mappings[col])
    
    buy_pred = model.predict(input_df)
    rent_pred = rent_model.predict(input_df)

    st.write(f"### Kupno: {buy_pred[0]:,.2f}zł")
    st.write(f"### Wynajem: {rent_pred[0]:,.2f}zł/miesiąc")

with t_input:
    file = st.file_uploader("Wgraj plik CSV z danymi mieszkań", type=["csv"])
    st.write("""Plik CSV powinien zawierać dokładnie takie kolumny:
- `id`: int
- `city`: [Białystok, Bydgoszcz, Częstochowa, Gdańsk, Gdynia, Katowice, Kraków, Łódź, Lublin, Poznań, Radom, Rzeszów, Szczecin, Warszawa, Wrocław]
- `type`: [apartmentBuilding, blockOfFlats, tenement]
- `squareMeters`: float
- `rooms`: int
- `floor`: int
- `floorCount`: int
- `buildYear`: int
- `centreDistance`: float
- `poiCount`: int
- `schoolDistance`: float
- `clinicDistance`: float
- `postOfficeDistance`: float
- `kindergartenDistance`: float
- `restaurantDistance`: float
- `collegeDistance`: float
- `pharmacyDistance`: float
- `ownership`: [condominium, cooperative, udział]
- `buildingMaterial`: [brick, concreteSlab]
- `condition`: [low, premium]
- `hasParkingSpace`: [yes, no]
- `hasBalcony`: [yes, no]
- `hasElevator`: [yes, no]
- `hasSecurity`: [yes, no]
- `hasStorageRoom`: [yes, no]
- `price`: float""")
    isRent = st.toggle("Trenuj model dla wynajmu mieszkań", value=False, help="Zaznacz, jeśli chcesz trenować model dla wynajmu mieszkań (domyślnie model jest trenowany dla kupna mieszkań).")
    le, mi, ri = st.columns(3)
    with le:
        if file: st.button("Trenuj model", on_click=retrain, args=(file, False), help="Trenuje model od nowa.", use_container_width=True)
        else: st.button("Trenuj model", disabled=True, help="Najpierw wgraj plik CSV z danymi mieszkań.", use_container_width=True)
    with mi:
        if file: st.button("Dotrenuj model", on_click=retrain, args=(file, True), help="Trenuje model na podstawie istniejącego modelu w pamięci.", use_container_width=True)
        else: st.button("Dotrenuj model", disabled=True, help="Najpierw wgraj plik CSV z danymi mieszkań.", use_container_width=True)
    with ri: st.button("Resetuj model", on_click=untrain, help="Przywraca model do stanu oryginalnego.", use_container_width=True)

with t_info:
    st.image("https://www.victoriadom.pl/static/a624fb2f6d84c6f8438b826ed71a4128/6f66c/co-to-jest-i-jak-funkcjonuje-wspolnota-mieszkaniowa.webp")

    st.subheader("Natan Majewski s25827")