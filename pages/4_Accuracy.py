import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

def show_classification_report(y_true, y_pred):
    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )
    df_report = pd.DataFrame(report_dict).transpose().round(2)

    # Clean accuracy row (optional but looks better)
    if "accuracy" in df_report.index:
        df_report.loc["accuracy", ["precision", "recall", "f1-score"]] = None

    st.subheader("Model Classification Report")
    st.dataframe(df_report, use_container_width=True)

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
        
        new_crimes_data = crimes_data.loc[(crimes_data['x_coordinate']!=0)]

        #Converting the numercial attributes to categorical attributes
        crimes_data.year = pd.Categorical(crimes_data.year)
        crimes_data.time = pd.Categorical(crimes_data.time)
        crimes_data.domestic = pd.Categorical(crimes_data.domestic)
        crimes_data.arrest = pd.Categorical(crimes_data.arrest)
        crimes_data.beat = pd.Categorical(crimes_data.beat)
        crimes_data.district = pd.Categorical(crimes_data.district)
        crimes_data.ward = pd.Categorical(crimes_data.ward)
        crimes_data.community_area = pd.Categorical(crimes_data.community_area_number)

        crimes_data_prediction = crimes_data.drop(['date','block','iucr','primary_type','description','location_description','fbicode','updated_on','x_coordinate','y_coordinate'],axis=1)
        crimes_data_prediction.head()
        crimes_data_prediction.info()

        crimes_data_prediction = pd.get_dummies(crimes_data_prediction,drop_first=True)
        crimes_data_prediction.head()
        
        #Algorithms
        #Train test split with a test set size of 30% of entire data
        X_train, X_test, y_train, y_test = train_test_split(crimes_data_prediction.drop(['arrest_1'],axis=1),crimes_data_prediction['arrest_1'], test_size=0.3, random_state=42)
        #Standardizing the data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        #Gaussain Naive Bayes
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)

        st.markdown("<h1 style='text-align: center;'>PREDICTION OF CRIME</h1>", unsafe_allow_html=True)
        add_vertical_space(5)
        st.header("Gaussian Naive Bayes")
        st.markdown("---")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt=".3f", square=True, cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)

        st.write('Accuracy = ',metrics.accuracy_score(y_test, y_pred))
        st.write('Error = ',1 - metrics.accuracy_score(y_test, y_pred))
        st.write('Precision = ',metrics.precision_score(y_test, y_pred,))
        st.write('Recall = ',metrics.recall_score(y_test, y_pred))
        st.write('F-1 Score = ',metrics.f1_score(y_test, y_pred))
        st.text('Confusion Matrix:')
        st.write(conf_matrix)
        add_vertical_space(1)
        # NEW: Display model report as a structured table
        show_classification_report(y_test, y_pred)
        st.markdown("---")
        add_vertical_space(5)

        #Decision tree with Entropy as attribute measure
        model = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Compute confusion matrix
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        st.header("Decision Tree")
        st.markdown("---")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt=".3f", square=True, cmap="Blues", ax=ax)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)

        #Classification Metrics
        st.write('Accuracy = ',metrics.accuracy_score(y_test, y_pred))
        st.write('Error = ',1 - metrics.accuracy_score(y_test, y_pred))
        st.write('Precision = ',metrics.precision_score(y_test, y_pred,))
        st.write('Recall = ',metrics.recall_score(y_test, y_pred))
        st.write('F-1 Score = ',metrics.f1_score(y_test, y_pred))
        st.text('Confusion Matrix:')
        st.write(conf_matrix)
        add_vertical_space(1)
        show_classification_report(y_test, y_pred)
        st.markdown("---")
        add_vertical_space(5)

        #Random Forest classifier  -
        model = RandomForestClassifier(n_estimators = 10,criterion='entropy',random_state=42)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        # Compute confusion matrix
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)

        #Plot confusion matrix
        st.header("Random Forest")
        st.markdown("---")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt=".3f", square=True, cmap="Blues", ax=ax)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion matrix')
        st.pyplot(fig)
        plt.close(fig)

        #Classification Metrics
        st.write('Accuracy = ',metrics.accuracy_score(y_test, y_pred))
        st.write('Error = ',1 - metrics.accuracy_score(y_test, y_pred))
        st.write('Precision = ',metrics.precision_score(y_test, y_pred,))
        st.write('Recall = ',metrics.recall_score(y_test, y_pred))
        st.write('F-1 Score = ',metrics.f1_score(y_test, y_pred))
        st.text('Confusion Matrix:')
        st.write(conf_matrix)
        add_vertical_space(1)
        show_classification_report(y_test, y_pred)
        st.markdown("---")
        add_vertical_space(5)

        #Logistic Regression
        classifier = LogisticRegression(random_state=42)
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        # Compute confusion matrix
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        st.header("Logistic Regression")
        st.markdown("---")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt=".3f", square=True, cmap="Blues", ax=ax)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion matrix')
        st.pyplot(fig)
        plt.close(fig)

        #Classification Metrics
        st.write('Accuracy = ',metrics.accuracy_score(y_test, y_pred))
        st.write('Error = ',1 - metrics.accuracy_score(y_test, y_pred))
        st.write('Precision = ',metrics.precision_score(y_test, y_pred,))
        st.write('Recall = ',metrics.recall_score(y_test, y_pred))
        st.write('F-1 Score = ',metrics.f1_score(y_test, y_pred))
        st.text('Confusion Matrix:')
        st.write(conf_matrix)
        add_vertical_space(1)
        show_classification_report(y_test, y_pred)
        st.markdown("---")
        add_vertical_space(5)
        
        #Predicting type of crime
        crimes_data_type = crimes_data.loc[crimes_data.primary_type_grouped.isin(['THEFT','NON-CRIMINAL_ASSAULT','CRIMINAL_OFFENSE'])]
        crimes_data_prediction = crimes_data_type.drop(['date','block','iucr','primary_type','description','location_description','fbicode','updated_on','x_coordinate','y_coordinate','primary_type_grouped'],axis=1)
        crimes_data_prediction_type = crimes_data_type.primary_type_grouped
        crimes_data_prediction = pd.get_dummies(crimes_data_prediction,drop_first=True)


        X_train, X_test, y_train, y_test = train_test_split(crimes_data_prediction,crimes_data_prediction_type, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        #Decision tree classifier for type of crime
        st.markdown("<h1 style='text-align: center;'>PREDICTION OF CRIME TYPE</h1>", unsafe_allow_html=True)
        add_vertical_space(1)
        st.header("Gaussian Naive Bayes")
        st.markdown("---")
        # Train model
        gnb_model = GaussianNB()
        gnb_model.fit(X_train, y_train)
        # Predict
        y_pred = gnb_model.predict(X_test)
        # Confusion matrix
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)
        # Metrics
        st.write('Accuracy = ', metrics.accuracy_score(y_test, y_pred))
        st.write('Error = ', 1 - metrics.accuracy_score(y_test, y_pred))
        st.text('Confusion Matrix:')
        st.write(conf_matrix)
        add_vertical_space(1)
        # Table-based classification report
        show_classification_report(y_test, y_pred)
        st.markdown("---")
        add_vertical_space(5)
        
        st.header("Decision Tree")
        st.markdown("---")
        model = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Compute confusion matrix
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)
        #Classification Metrics
        st.write('Accuracy = ',metrics.accuracy_score(y_test, y_pred))
        st.write('Error = ',1 - metrics.accuracy_score(y_test, y_pred))
        st.text('Confusion Matrix:')
        st.write(conf_matrix)
        add_vertical_space(1)
        show_classification_report(y_test, y_pred)
        st.markdown("---")
        add_vertical_space(5)

        #Random Forest classifier for type of crime
        st.header("Random Forest")
        st.markdown("---")
        model = RandomForestClassifier(n_estimators = 10,criterion='entropy',random_state=42)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        # Compute confusion matrix
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)
        #Classification Metrics
        st.write('Accuracy = ',metrics.accuracy_score(y_test, y_pred))
        st.write('Error = ',1 - metrics.accuracy_score(y_test, y_pred))
        st.text('Confusion Matrix:')
        st.write(conf_matrix)
        add_vertical_space(1)
        show_classification_report(y_test, y_pred)
        st.markdown("---")
        add_vertical_space(5)

        #Logistic Regression for predicting the type of crime -Best
        st.header("Logistic Regression")
        st.markdown("---")
        classifier = LogisticRegression(random_state=42)
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        # Compute confusion matrix
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)
        st.write('Accuracy = ',metrics.accuracy_score(y_test, y_pred))
        st.write('Error = ',1 - metrics.accuracy_score(y_test, y_pred))
        st.text('Confusion Matrix:')
        st.write(conf_matrix)
        add_vertical_space(1)
        show_classification_report(y_test, y_pred)
        st.markdown("---")
        add_vertical_space(5)
       