import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

# Load your data
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    
    if st.sidebar.button("Show Analysis"):
    # Read the uploaded file into a pandas DataFrame
        crimes_data = pd.read_csv(uploaded_file)
        
        #handling inconsistency in column names
        crimes_data.columns = crimes_data.columns.str.strip()
        crimes_data.columns = crimes_data.columns.str.replace(',', '')
        crimes_data.columns = crimes_data.columns.str.replace(' ', '_')
        crimes_data.columns = crimes_data.columns.str.lower()
        crimes_data.duplicated(keep=False)
        
        # Removing Primary key type attributes as they are of no use for any type of analysis, Location columns is just a combination of Latitude and Longitude
        crimes_data.drop(['id','case_number','location'],axis=1,inplace=True)
        
        #Dropping observations where latitude is null/Nan
        crimes_data.dropna(subset=['latitude'],inplace=True)
        crimes_data.reset_index(drop=True,inplace=True)
        crimes_data.isnull().sum()
        crimes_data.dropna(inplace=True)
        crimes_data.reset_index(drop=True,inplace=True)
        crimes_data.info()
        
        #Converting the data column to datetime object so we can get better results of our analysis
        #Get the day of the week,month and time of the crimes
        crimes_data.date = pd.to_datetime(crimes_data.date)
        crimes_data['day_of_week'] = crimes_data.date.dt.day_name()
        crimes_data['month'] = crimes_data.date.dt.month_name()
        crimes_data['time'] = crimes_data.date.dt.hour
        
        #Mapping similar crimes under one group.
        primary_type_map = {
            ('BURGLARY','MOTOR VEHICLE THEFT','THEFT','ROBBERY') : 'THEFT',
            ('BATTERY','ASSAULT','NON-CRIMINAL','NON-CRIMINAL (SUBJECT SPECIFIED)') : 'NON-CRIMINAL_ASSAULT',
            ('CRIM SEXUAL ASSAULT','SEX OFFENSE','STALKING','PROSTITUTION') : 'SEXUAL_OFFENSE',
            ('WEAPONS VIOLATION','CONCEALED CARRY LICENSE VIOLATION') :  'WEAPONS_OFFENSE',
            ('HOMICIDE','CRIMINAL DAMAGE','DECEPTIVE PRACTICE','CRIMINAL TRESPASS') : 'CRIMINAL_OFFENSE',
            ('KIDNAPPING','HUMAN TRAFFICKING','OFFENSE INVOLVING CHILDREN') : 'HUMAN_TRAFFICKING_OFFENSE',
            ('NARCOTICS','OTHER NARCOTIC VIOLATION') : 'NARCOTIC_OFFENSE',
            ('OTHER OFFENSE','ARSON','GAMBLING','PUBLIC PEACE VIOLATION','INTIMIDATION','INTERFERENCE WITH PUBLIC OFFICER','LIQUOR LAW VIOLATION','OBSCENITY','PUBLIC INDECENCY') : 'OTHER_OFFENSE'
        }
        primary_type_mapping = {}
        for keys, values in primary_type_map.items():
            for key in keys:
                primary_type_mapping[key] = values
        crimes_data['primary_type_grouped'] = crimes_data.primary_type.map(primary_type_mapping)

        #Zone where the crime has occured
        zone_mapping = {
            'N' : 'North',
            'S' : 'South',
            'E' : 'East',
            'W' : 'West'
        }
        crimes_data['zone'] = crimes_data.block.str.split(" ", n = 2, expand = True)[1].map(zone_mapping)
        #Mapping seasons from month of crime
        season_map = {
            ('March','April','May') : 'Spring',
            ('June','July','August') : 'Summer',
            ('September','October','November') : 'Fall',
            ('December','January','February') : 'Winter'
        }
        season_mapping = {}
        for keys, values in season_map.items():
            for key in keys:
                season_mapping[key] = values
        crimes_data['season'] = crimes_data.month.map(season_mapping)

        #Mapping similar locations of crime under one group.
        loc_map = {
            ('RESIDENCE', 'APARTMENT', 'CHA APARTMENT', 'RESIDENCE PORCH/HALLWAY', 'RESIDENCE-GARAGE',
            'RESIDENTIAL YARD (FRONT/BACK)', 'DRIVEWAY - RESIDENTIAL', 'HOUSE') : 'RESIDENCE',

            ('BARBERSHOP', 'COMMERCIAL / BUSINESS OFFICE', 'CURRENCY EXCHANGE', 'DEPARTMENT STORE', 'RESTAURANT',
            'ATHLETIC CLUB', 'TAVERN/LIQUOR STORE', 'SMALL RETAIL STORE', 'HOTEL/MOTEL', 'GAS STATION',
            'AUTO / BOAT / RV DEALERSHIP', 'CONVENIENCE STORE', 'BANK', 'BAR OR TAVERN', 'DRUG STORE',
            'GROCERY FOOD STORE', 'CAR WASH', 'SPORTS ARENA/STADIUM', 'DAY CARE CENTER', 'MOVIE HOUSE/THEATER',
            'APPLIANCE STORE', 'CLEANING STORE', 'PAWN SHOP', 'FACTORY/MANUFACTURING BUILDING', 'ANIMAL HOSPITAL',
            'BOWLING ALLEY', 'SAVINGS AND LOAN', 'CREDIT UNION', 'KENNEL', 'GARAGE/AUTO REPAIR', 'LIQUOR STORE',
            'GAS STATION DRIVE/PROP.', 'OFFICE', 'BARBER SHOP/BEAUTY SALON') : 'BUSINESS',

            ('VEHICLE NON-COMMERCIAL', 'AUTO', 'VEHICLE - OTHER RIDE SHARE SERVICE (E.G., UBER, LYFT)', 'TAXICAB',
            'VEHICLE-COMMERCIAL', 'VEHICLE - DELIVERY TRUCK', 'VEHICLE-COMMERCIAL - TROLLEY BUS',
            'VEHICLE-COMMERCIAL - ENTERTAINMENT/PARTY BUS') : 'VEHICLE',

            ('AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA', 'CTA PLATFORM', 'CTA STATION', 'CTA BUS STOP',
            'AIRPORT TERMINAL UPPER LEVEL - SECURE AREA', 'CTA TRAIN', 'CTA BUS', 'CTA GARAGE / OTHER PROPERTY',
            'OTHER RAILROAD PROP / TRAIN DEPOT', 'AIRPORT TERMINAL LOWER LEVEL - SECURE AREA',
            'AIRPORT BUILDING NON-TERMINAL - SECURE AREA', 'AIRPORT EXTERIOR - NON-SECURE AREA', 'AIRCRAFT',
            'AIRPORT PARKING LOT', 'AIRPORT TERMINAL LOWER LEVEL - NON-SECURE AREA', 'OTHER COMMERCIAL TRANSPORTATION',
            'AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA', 'AIRPORT VENDING ESTABLISHMENT',
            'AIRPORT TERMINAL MEZZANINE - NON-SECURE AREA', 'AIRPORT EXTERIOR - SECURE AREA', 'AIRPORT TRANSPORTATION SYSTEM (ATS)',
            'CTA TRACKS - RIGHT OF WAY', 'AIRPORT/AIRCRAFT', 'BOAT/WATERCRAFT', 'CTA PROPERTY', 'CTA "L" PLATFORM',
            'RAILROAD PROPERTY') : 'PUBLIC_TRANSPORTATION',

            ('HOSPITAL BUILDING/GROUNDS', 'NURSING HOME/RETIREMENT HOME', 'SCHOOL, PUBLIC, BUILDING',
            'CHURCH/SYNAGOGUE/PLACE OF WORSHIP', 'SCHOOL, PUBLIC, GROUNDS', 'SCHOOL, PRIVATE, BUILDING',
            'MEDICAL/DENTAL OFFICE', 'LIBRARY', 'COLLEGE/UNIVERSITY RESIDENCE HALL', 'YMCA', 'HOSPITAL') : 'PUBLIC_BUILDING',

            ('STREET', 'PARKING LOT/GARAGE(NON.RESID.)', 'SIDEWALK', 'PARK PROPERTY', 'ALLEY', 'CEMETARY',
            'CHA HALLWAY/STAIRWELL/ELEVATOR', 'CHA PARKING LOT/GROUNDS', 'COLLEGE/UNIVERSITY GROUNDS', 'BRIDGE',
            'SCHOOL, PRIVATE, GROUNDS', 'FOREST PRESERVE', 'LAKEFRONT/WATERFRONT/RIVERBANK', 'PARKING LOT', 'DRIVEWAY',
            'HALLWAY', 'YARD', 'CHA GROUNDS', 'RIVER BANK', 'STAIRWELL', 'CHA PARKING LOT') : 'PUBLIC_AREA',

            ('POLICE FACILITY/VEH PARKING LOT', 'GOVERNMENT BUILDING/PROPERTY', 'FEDERAL BUILDING', 'JAIL / LOCK-UP FACILITY',
            'FIRE STATION', 'GOVERNMENT BUILDING') : 'GOVERNMENT',

            ('OTHER', 'ABANDONED BUILDING', 'WAREHOUSE', 'ATM (AUTOMATIC TELLER MACHINE)', 'VACANT LOT/LAND',
            'CONSTRUCTION SITE', 'POOL ROOM', 'NEWSSTAND', 'HIGHWAY/EXPRESSWAY', 'COIN OPERATED MACHINE', 'HORSE STABLE',
            'FARM', 'GARAGE', 'WOODED AREA', 'GANGWAY', 'TRAILER', 'BASEMENT', 'CHA PLAY LOT') : 'OTHER'
        }

        loc_mapping = {}
        for keys, values in loc_map.items():
            for key in keys:
                loc_mapping[key] = values
        crimes_data['loc_grouped'] = crimes_data.location_description.map(loc_mapping)

        #Mapping crimes to ints to get better information from plots
        crimes_data.arrest = crimes_data.arrest.astype(int)
        crimes_data.domestic = crimes_data.domestic.astype(int)
        
        #zones plots
        #CrimeWise
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.markdown("<h1 style='text-align: center;'>CRIME ZONES</h1>", unsafe_allow_html=True)
        add_vertical_space(2)
        st.header("Crime Zone Wise")
        st.caption("This is a count graph that displays crimes counts zones wise like South, North, East and West. This graph will provide with the information where the crimes are mostly happening in the area.")
        zone_plot = sns.countplot(data=crimes_data,x='zone',order=crimes_data.zone.value_counts().index,palette='Set2')
        st.pyplot()
        
        #Crime Zone Year Wise
        add_vertical_space(10)
        st.header("Crime Zone Year Wise")
        st.caption("This is a statistical bar graph that shows the number of crimes committed in particular zones over the past years. This information will help us in acknowledging whether there is a dip or increase in the crime rate.")
        plt.figure(figsize=(15,10))
        zone_plot = sns.countplot(data=crimes_data,x='zone',hue='year',order=crimes_data.zone.value_counts().index,palette='Set2')
        st.pyplot()

        #Crime Season Wise
        add_vertical_space(10)
        st.header("Crime Season Wise")
        st.caption("This is a visual display on how crimes are occuring over the past years and also providing us with the information that in which seasons are crime rate are at peak. The seasons are categorised as Spring, Summer, Winter and Autumn.")
        plt.figure(figsize=(15, 10))
        zone_plot = sns.countplot(data=crimes_data,x='season',hue='year',palette='Set2')
        st.pyplot()