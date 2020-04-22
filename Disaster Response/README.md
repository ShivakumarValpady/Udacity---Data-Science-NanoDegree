# Disaster Response Pipeline Project

### Motivation
The purpose of this project is to make a web app using a both an ETL and Machine Learning Pipelines to create a model that will send messages to a specific disaster relief organization.

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.
   - To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
   - To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

